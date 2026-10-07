#!/usr/bin/env python3
"""Decision test harness for a live Clef ``POST /v1/systemone`` endpoint.

A suite file describes one workflow: the question schema production sends plus labeled cases.
Each case is sent as one request and the returned decisions are checked against its expectations.

    kubectl port-forward svc/clef-server 8000:8000 &        # from a workstation
    python tests/decisions/run_decisions.py tests/decisions/suites/*.yaml
    python tests/decisions/run_decisions.py suite.yaml -v -k refund --tag billing --out results/decisions.json
    python run_decisions.py suite.yaml --url http://clef-server:8000        # in-cluster (clef-bench-client)

Suite file (YAML, or the same structure as JSON):

    name: support-triage
    model: clef                       # optional; sent as the request "model"
    min_pass_rate: 1.0                # optional; share of graded cases that must pass
    questions:                        # the schema exactly as production sends it (same order too):
      department:                     # Clef answers all questions jointly, so test the real schema
        type: choice
        instructions: Which team should handle the message?
        criteria: {billing: Payments or invoices, technical: Bugs or outages}
      urgency: {type: score, criteria: [Can wait, This week, Today]}
      outage: {type: noul, instructions: Is a service down?}
    defaults: {min_confidence: 0.6}   # optional: min/max_confidence, threshold, tolerance for every case
    cases:
      - id: checkout-down
        state: Our checkout started returning errors and orders are blocked.  # text or any JSON value
        tags: [outage]
        expect:                       # shorthand, by question type:
          department: technical       #   choice: the expected option, or a list of acceptable options
          urgency: 2                  #   score: expected score within +/-0.5 (rounds to that level)
          outage: true                #   noul: P(true) >= 0.5 for true, < 0.5 for false
      - id: invoice-question
        state: {customer: Acme, message: Why was I charged twice this month?}
        expect:
          department: {choice: billing, min_confidence: 0.9}
          urgency: {max: 1.5}
        xfail: tracked in BUG-123     # known failure: reported, but excluded from the pass rate

Expectation keys (dict form):
    choice: option | [options]      not_choice: option | [options]                    (choice)
    noul: true | false              threshold: P(true) cut-off, default 0.5           (noul)
    score: value                    tolerance: allowed |score - value|, default 0.5   (score)
    level: index of the most likely level                                            (score)
    min / max: bounds on P(true) (noul) or on the expected score (score)
    min_confidence / max_confidence: bounds on the top probability (noul: max(p, 1 - p))

A case may also set ``questions`` (merged over the suite schema; null drops a question) and
``images`` (base64 strings). Unknown keys, unknown question ids and options missing from the
schema are rejected before any request is sent, so a typo cannot pass silently.

Exit status: 0 if every suite meets its pass rate, 1 if not, 2 on setup errors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

TYPE_KEYS = {
    "choice": {"choice", "not_choice"},
    "noul": {"noul", "threshold", "min", "max"},
    "score": {"score", "tolerance", "level", "min", "max"},
}
COMMON_KEYS = {"min_confidence", "max_confidence"}
NUMERIC_KEYS = {"score", "threshold", "tolerance", "min", "max", "min_confidence", "max_confidence"}
MODIFIER_KEYS = {"threshold", "tolerance"}  # adjust a check but check nothing on their own
DEFAULT_KEYS = COMMON_KEYS | MODIFIER_KEYS  # allowed in a suite's `defaults`
SUITE_KEYS = {"name", "model", "min_pass_rate", "questions", "defaults", "cases"}
CASE_KEYS = {"id", "state", "expect", "tags", "xfail", "questions", "images"}
GRADED = ("pass", "fail", "error")  # xfail / xpass are reported but excluded from the pass rate
STATUS_STYLE = {"pass": ("PASS ", "32"), "fail": ("FAIL ", "31"), "error": ("ERROR", "1;31"),
                "xfail": ("XFAIL", "33"), "xpass": ("XPASS", "36")}


class SuiteError(ValueError):
    """Invalid suite file. Raised while loading, before any request is sent."""


# --------------------------------------------------------------------------------------
# Suite loading and validation
# --------------------------------------------------------------------------------------


def _json_key(key: Any) -> str:
    """The string a value becomes as a JSON object key (True -> "true", 1 -> "1")."""
    return key if isinstance(key, str) else json.dumps(key)


def _wire(value: Any, ctx: str) -> Any:
    """Round-trip through JSON: keys become the exact strings the server sees, YAML dates become text."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError) as exc:
        raise SuiteError(f"{ctx}: not representable as JSON ({exc})") from None


def _load_yaml(text: str) -> Any:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    # YAML 1.1 reads yes/no/on/off as booleans; keep only true/false so option ids like "yes" stay strings.
    bool_tag = "tag:yaml.org,2002:bool"
    Loader.yaml_implicit_resolvers = {
        ch: [r for r in rs if r[0] != bool_tag] for ch, rs in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    Loader.add_implicit_resolver(bool_tag, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
    return yaml.load(text, Loader=Loader)


def _no_unknown(obj: Any, allowed: set[str], ctx: str) -> None:
    if not isinstance(obj, dict):
        raise SuiteError(f"{ctx}: expected a mapping, got {type(obj).__name__}")
    unknown = sorted(str(k) for k in obj if k not in allowed)
    if unknown:
        raise SuiteError(f"{ctx}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def load_suite(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        data = _load_yaml(text) if path.endswith((".yaml", ".yml")) else json.loads(text)
    except ImportError:
        raise SuiteError(f"{path}: YAML suites need PyYAML (pip install pyyaml); JSON suites need nothing") from None
    except Exception as exc:  # noqa: BLE001  (missing file or parse error)
        raise SuiteError(f"{path}: {exc}") from None
    return validate_suite(data, path)


def validate_suite(data: Any, path: str) -> dict[str, Any]:
    where = os.path.basename(path)
    _no_unknown(data, SUITE_KEYS, where)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SuiteError(f"{where}: 'cases' must be a non-empty list")
    defaults = data.get("defaults") or {}
    _no_unknown(defaults, DEFAULT_KEYS, f"{where}: defaults")
    suite_questions = data.get("questions") or {}
    if not isinstance(suite_questions, dict):
        raise SuiteError(f"{where}: 'questions' must map question ids to questions")
    rate = data.get("min_pass_rate", 1.0)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        raise SuiteError(f"{where}: min_pass_rate must be a number from 0 to 1")

    out, seen = [], set()
    for i, case in enumerate(cases, 1):
        cid = str(case.get("id", f"case-{i}")) if isinstance(case, dict) else f"case-{i}"
        ctx = f"{where}:{cid}"
        _no_unknown(case, CASE_KEYS, ctx)
        if cid in seen:
            raise SuiteError(f"{ctx}: duplicate case id")
        seen.add(cid)
        if "state" not in case:
            raise SuiteError(f"{ctx}: 'state' is required")
        overrides = case.get("questions") or {}
        if not isinstance(overrides, dict):
            raise SuiteError(f"{ctx}: 'questions' must map question ids to questions")
        questions = _wire({k: v for k, v in {**suite_questions, **overrides}.items() if v is not None}, ctx)
        _check_questions(questions, ctx)
        expect = case.get("expect")
        if not isinstance(expect, dict) or not expect:
            raise SuiteError(f"{ctx}: 'expect' must map question ids to expectations")
        checks = {}
        for qid, spec in expect.items():
            qid = _json_key(qid)
            if qid not in questions:
                raise SuiteError(f"{ctx}: expectation for unknown question '{qid}' (schema: {sorted(questions)})")
            checks[qid] = normalize_expectation(spec, questions[qid], defaults, f"{ctx}.{qid}")
        tags = case.get("tags") or []
        out.append(dict(
            id=cid, state=_wire(case["state"], ctx), questions=questions, checks=checks, images=case.get("images"),
            tags=[str(t) for t in ([tags] if isinstance(tags, str) else tags)],
            xfail=str(case["xfail"]) if case.get("xfail") else None,
        ))
    return dict(name=str(data.get("name") or os.path.splitext(where)[0]), model=str(data.get("model") or "clef"),
                min_pass_rate=float(rate), path=path, cases=out)


def _check_questions(questions: dict[str, Any], ctx: str) -> None:
    if not questions:
        raise SuiteError(f"{ctx}: no questions; set 'questions' on the suite or the case")
    for qid, q in questions.items():
        qtype = q.get("type") if isinstance(q, dict) else None
        if qtype not in TYPE_KEYS:
            raise SuiteError(f"{ctx}: question '{qid}' needs type noul, choice or score")
        criteria = q.get("criteria")
        if qtype == "choice" and not (isinstance(criteria, dict) and criteria):
            raise SuiteError(f"{ctx}: choice question '{qid}' needs a non-empty criteria mapping")
        if qtype == "score" and not (isinstance(criteria, list) and criteria):
            raise SuiteError(f"{ctx}: score question '{qid}' needs a non-empty criteria list")


def normalize_expectation(spec: Any, question: dict[str, Any], defaults: dict[str, Any], ctx: str) -> dict[str, Any]:
    qtype = question["type"]
    allowed = TYPE_KEYS[qtype] | COMMON_KEYS
    if not isinstance(spec, dict):
        spec = {qtype: spec}  # shorthand: the main check is named after the question type
    _no_unknown(spec, allowed, f"{ctx} ({qtype} question)")
    spec = {**{k: v for k, v in defaults.items() if k in allowed}, **spec}
    for key in sorted(NUMERIC_KEYS & set(spec)):
        if isinstance(spec[key], bool) or not isinstance(spec[key], (int, float)):
            raise SuiteError(f"{ctx}: {key} must be a number, got {spec[key]!r}")
    if qtype == "choice":
        options = list(question["criteria"])
        for key in ("choice", "not_choice"):
            if key in spec:
                values = [_json_key(v) for v in (spec[key] if isinstance(spec[key], list) else [spec[key]])]
                missing = [v for v in values if v not in options]
                if not values or missing:
                    raise SuiteError(f"{ctx}: {key} {missing or values} must be among the options {options}")
                spec[key] = values
    elif qtype == "noul" and "noul" in spec:
        value = spec["noul"]
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if not isinstance(value, bool):
            raise SuiteError(f"{ctx}: expected true or false, got {value!r}")
        spec["noul"] = value
    elif qtype == "score":
        top = len(question["criteria"]) - 1
        if "score" in spec and not 0 <= spec["score"] <= top:
            raise SuiteError(f"{ctx}: score {spec['score']} is outside 0..{top}")
        if "level" in spec and spec["level"] not in range(top + 1):
            raise SuiteError(f"{ctx}: level {spec['level']!r} is outside 0..{top}")
    if not set(spec) - MODIFIER_KEYS:
        raise SuiteError(f"{ctx}: nothing to check")
    return spec


# --------------------------------------------------------------------------------------
# Checking answers
# --------------------------------------------------------------------------------------


def _level(ans: dict[str, Any]) -> int:
    probs = ans.get("probabilities") or {}
    return int(max(probs, key=probs.get)) if probs else round(float(ans["score"]))


def describe(ans: dict[str, Any]) -> str:
    """One-line summary of a SystemOne answer."""
    kind = ans.get("type")
    if kind == "choice":
        return f"{ans['choice']} ({ans['confidence']:.2f})"
    if kind == "noul":
        return f"P(true)={ans['noul']:.3f}"
    if kind == "score":
        return f"score={ans['score']:.2f} (level {_level(ans)}, {ans['confidence']:.2f})"
    return json.dumps(ans)


def distribution(ans: dict[str, Any], top: int = 4) -> str:
    """Score levels in order (if few), otherwise the most likely options first."""
    probs = ans.get("probabilities") or {}
    if ans.get("type") == "score" and len(probs) <= 6:
        items = list(probs.items())
    else:
        items = sorted(probs.items(), key=lambda kv: -kv[1])[:top]
    return " ".join(f"{k}={v:.2f}" for k, v in items)


def evaluate(qtype: str, spec: dict[str, Any], ans: dict[str, Any]) -> tuple[list[str], float]:
    """Check one answer against one expectation. Returns (failure messages, confidence)."""
    fails: list[str] = []
    got = describe(ans)
    if qtype == "choice":
        choice, conf = str(ans["choice"]), float(ans["confidence"])
        if "choice" in spec and choice not in spec["choice"]:
            fails.append(f"expected {' | '.join(spec['choice'])}, got {got}")
        if "not_choice" in spec and choice in spec["not_choice"]:
            fails.append(f"must not be {got}")
    elif qtype == "noul":
        p = float(ans["noul"])
        conf, thr = max(p, 1.0 - p), spec.get("threshold", 0.5)
        if "noul" in spec and (p >= thr) != spec["noul"]:
            fails.append(f"expected {'true' if spec['noul'] else 'false'}, got {got} (threshold {thr})")
        if "min" in spec and p < spec["min"]:
            fails.append(f"{got} below min {spec['min']}")
        if "max" in spec and p > spec["max"]:
            fails.append(f"{got} above max {spec['max']}")
    else:
        score, conf, tol = float(ans["score"]), float(ans["confidence"]), spec.get("tolerance", 0.5)
        if "score" in spec and abs(score - spec["score"]) > tol + 1e-9:
            fails.append(f"expected {spec['score']} +/- {tol}, got {got}")
        if "level" in spec and _level(ans) != spec["level"]:
            fails.append(f"expected level {spec['level']}, got {got}")
        if "min" in spec and score < spec["min"]:
            fails.append(f"{got} below min {spec['min']}")
        if "max" in spec and score > spec["max"]:
            fails.append(f"{got} above max {spec['max']}")
    if "min_confidence" in spec and conf < spec["min_confidence"]:
        fails.append(f"got {got}, confidence {conf:.2f} below min {spec['min_confidence']}")
    if "max_confidence" in spec and conf > spec["max_confidence"]:
        fails.append(f"got {got}, confidence {conf:.2f} above max {spec['max_confidence']}")
    return fails, conf


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    xfail: str | None
    status: str = "pass"  # pass | fail | error | xfail (failed as expected) | xpass (xfail case that passed)
    latency_ms: float = 0.0
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)  # question id -> outcome of its expectation
    answers: dict[str, Any] = field(default_factory=dict)  # every answer returned, checked or not
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def http_json(url: str, body: Any = None, timeout: float = 120.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from None


def run_case(url: str, suite: dict[str, Any], case: dict[str, Any], timeout: float) -> CaseResult:
    res = CaseResult(case["id"], case["tags"], case["xfail"])
    body = {"model": suite["model"], "state": case["state"], "questions": case["questions"]}
    if case["images"]:
        body["images"] = case["images"]
    t0 = time.perf_counter()
    try:
        resp = http_json(url + "/v1/systemone", body, timeout)
        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.answers = resp.get("answers") or {}
        res.warnings = list(resp.get("warnings") or [])
        for qid, spec in case["checks"].items():
            qtype = case["questions"][qid]["type"]
            if qid not in res.answers:
                res.checks[qid] = dict(type=qtype, passed=False, failures=["missing from the response"], expected=spec)
                continue
            failures, confidence = evaluate(qtype, spec, res.answers[qid])
            res.checks[qid] = dict(type=qtype, passed=not failures, failures=failures, confidence=confidence,
                                   expected=spec)
    except Exception as exc:  # noqa: BLE001  (network, HTTP or malformed response: reported on the case)
        res.status, res.error, res.checks = "error", f"{type(exc).__name__}: {exc}", {}
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res
    failed = any(not c["passed"] for c in res.checks.values())
    if res.xfail:
        res.status = "xfail" if failed else "xpass"
    else:
        res.status = "fail" if failed else "pass"
    return res


def run_suite(url: str, suite: dict[str, Any], concurrency: int, timeout: float) -> list[CaseResult]:
    cases = suite["cases"]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_case, url, suite, case, timeout) for case in cases]
        if sys.stderr.isatty():
            for done, _ in enumerate(as_completed(futures), 1):
                print(f"\r  {suite['name']}: {done}/{len(futures)}", end="", file=sys.stderr, flush=True)
            print("\r\033[K", end="", file=sys.stderr, flush=True)
        return [f.result() for f in futures]


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _pct(sorted_xs: list[float], q: float) -> float | None:
    return sorted_xs[min(len(sorted_xs) - 1, round(q / 100 * (len(sorted_xs) - 1)))] if sorted_xs else None


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return "-" if x is None else format(x, spec)


def summarize(results: list[CaseResult], min_pass_rate: float) -> dict[str, Any]:
    counts = Counter(r.status for r in results)
    graded = sum(counts[s] for s in GRADED)
    pass_rate = counts["pass"] / graded if graded else 1.0
    per_q: dict[str, dict[str, Any]] = {}
    confusions: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for qid, c in r.checks.items():
            q = per_q.setdefault(qid, dict(type=c["type"], passed=0, total=0, conf_pass=[], conf_fail=[]))
            q["total"] += 1
            q["passed"] += int(c["passed"])
            if "confidence" in c:
                (q["conf_pass"] if c["passed"] else q["conf_fail"]).append(c["confidence"])
            want = c["expected"].get("choice") if c["type"] == "choice" else None
            got = r.answers.get(qid, {}).get("choice")
            if want and len(want) == 1 and got is not None and got != want[0]:
                confusions[qid][f"{want[0]} -> {got}"] += 1
    latencies = sorted(r.latency_ms for r in results if r.status != "error")
    return dict(
        cases=len(results), **{s: counts[s] for s in STATUS_STYLE},
        pass_rate=pass_rate, min_pass_rate=min_pass_rate, passed=pass_rate >= min_pass_rate - 1e-9,
        questions={qid: dict(type=q["type"], passed=q["passed"], total=q["total"], accuracy=q["passed"] / q["total"],
                             mean_conf_pass=_mean(q["conf_pass"]), mean_conf_fail=_mean(q["conf_fail"]))
                   for qid, q in per_q.items()},
        confusions={qid: dict(c.most_common()) for qid, c in confusions.items()},
        latency_ms=dict(p50=_pct(latencies, 50), p95=_pct(latencies, 95), max=latencies[-1] if latencies else None),
    )


def print_suite(suite: dict[str, Any], results: list[CaseResult], summary: dict[str, Any], verbose: bool,
                paint: Callable[[str, str], str]) -> None:
    print(f"\n{paint(suite['name'], '1')}  {suite['path']}  ({len(results)} cases)")
    for r in results:
        if r.status == "pass" and not verbose:
            continue
        label, color = STATUS_STYLE[r.status]
        tags = f"  [{', '.join(r.tags)}]" if r.tags else ""
        print(f"  {paint(label, color)} {r.id}{tags}  {r.latency_ms:.0f} ms")
        if r.status == "error":
            print(f"        {r.error}")
            continue
        if r.xfail:
            print(f"        xfail: {r.xfail}")
        for qid, ans in r.answers.items():
            check = r.checks.get(qid)
            if check and not check["passed"]:
                for msg in check["failures"]:
                    print(f"        {qid}: {paint(msg, '31')}")
                if ans.get("probabilities"):
                    print(f"        {' ' * len(qid)}  probabilities: {distribution(ans)}")
            elif verbose:
                print(f"        {qid}: {describe(ans)}" + ("" if check else "  (not checked)"))
        for qid, check in r.checks.items():
            if qid not in r.answers:
                print(f"        {qid}: {paint('; '.join(check['failures']), '31')}")
        for warning in r.warnings:
            print(f"        warning: {warning}")

    qs = summary["questions"]
    if qs:
        w = max(8, *(len(q) for q in qs))
        print(f"  {'question':<{w}}  {'type':<6}  {'passed':>7}  {'rate':>6}  {'conf ok':>7}  {'conf fail':>9}")
        for qid, q in qs.items():
            frac = f"{q['passed']}/{q['total']}"
            print(f"  {qid:<{w}}  {q['type']:<6}  {frac:>7}  {q['accuracy']:>6.1%}"
                  f"  {_fmt(q['mean_conf_pass']):>7}  {_fmt(q['mean_conf_fail']):>9}")
    for qid, pairs in summary["confusions"].items():
        print(f"  confusions in {qid} (expected -> got): " + ", ".join(f"{k} x{n}" for k, n in pairs.items()))
    s, lat = summary, summary["latency_ms"]
    verdict = paint("PASS", "32") if s["passed"] else paint("FAIL", "31")
    print(f"  {verdict}  {s['cases']} cases: {s['pass']} passed, {s['fail']} failed, {s['error']} errors, "
          f"{s['xfail']} xfail, {s['xpass']} xpass | pass rate {s['pass_rate']:.1%} (need {s['min_pass_rate'] * 100:g}%)"
          f" | latency p50 {_fmt(lat['p50'], '.0f')} ms, p95 {_fmt(lat['p95'], '.0f')} ms")


def _painter(enabled: bool) -> Callable[[str, str], str]:
    if not enabled:
        return lambda text, code: text
    return lambda text, code: f"\033[{code}m{text}\033[0m"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Clef decision suites against a live /v1/systemone endpoint.",
                                 epilog="Suite file format: see the docstring at the top of this file.")
    ap.add_argument("suites", nargs="+", help="suite files (.yaml, .yml or .json)")
    ap.add_argument("--url", default=os.environ.get("CLEF_URL", "http://localhost:8000"),
                    help="server base URL (default: $CLEF_URL or http://localhost:8000)")
    ap.add_argument("-k", dest="keyword", default="", help="only run cases whose id contains this text")
    ap.add_argument("--tag", action="append", default=[], help="only run cases with this tag (repeatable)")
    ap.add_argument("-c", "--concurrency", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds (default 120)")
    ap.add_argument("--min-pass-rate", type=float, help="override every suite's min_pass_rate (0 to 1)")
    ap.add_argument("--out", help="also write a JSON report (every answer, check and summary) to this path")
    ap.add_argument("-v", "--verbose", action="store_true", help="list passing cases and every decision")
    args = ap.parse_args(argv)
    url = args.url.rstrip("/")
    paint = _painter(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))

    try:
        suites = [load_suite(p) for p in args.suites]
    except SuiteError as exc:
        print(f"suite error: {exc}", file=sys.stderr)
        return 2
    for suite in suites:
        suite["cases"] = [c for c in suite["cases"] if args.keyword.lower() in c["id"].lower()
                          and (not args.tag or set(args.tag) & set(c["tags"]))]
    if not any(suite["cases"] for suite in suites):
        print("no cases match the -k / --tag filters", file=sys.stderr)
        return 2

    try:
        health = http_json(url + "/health", timeout=10)
        if health.get("status") != "ok":
            raise RuntimeError(f"/health returned {health}")
    except Exception as exc:  # noqa: BLE001
        print(f"server not ready at {url}: {exc}\n"
              f"  from a workstation: kubectl port-forward svc/clef-server 8000:8000\n"
              f"  in-cluster:         --url http://clef-server:8000", file=sys.stderr)
        return 2
    try:
        config = http_json(url + "/stats", timeout=10).get("config", {})
    except Exception:  # noqa: BLE001  (other SystemOne-compatible servers may not expose /stats)
        config = {}
    cfg = ", ".join(f"{k}={config[k]}" for k in ("quant", "fused", "attn_impl", "max_length") if k in config)
    print(f"Clef decision tests against {url}" + (f"  [{cfg}]" if cfg else ""))

    report, all_passed = [], True
    for suite in suites:
        if not suite["cases"]:
            continue
        results = run_suite(url, suite, args.concurrency, args.timeout)
        need = args.min_pass_rate if args.min_pass_rate is not None else suite["min_pass_rate"]
        summary = summarize(results, need)
        all_passed &= summary["passed"]
        print_suite(suite, results, summary, args.verbose, paint)
        report.append(dict(name=suite["name"], path=suite["path"], summary=summary,
                           cases=[asdict(r) for r in results]))

    print("\nRESULT: " + (paint("PASS", "32") if all_passed else paint("FAIL", "31")))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), url=url,
                           server_config=config, passed=all_passed, suites=report), f, indent=1)
        print(f"report: {args.out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
