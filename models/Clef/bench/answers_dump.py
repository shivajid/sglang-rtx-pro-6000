"""Send workload requests one at a time with debug=true and save per-question logits.

Used to check that two server configurations give the same answers (e.g. PAD_MULTIPLE=1 vs 16).
  python answers_dump.py <url> <workload.jsonl> <n> <out.json>
  python answers_dump.py --compare a.json b.json
"""

import json
import math
import sys
import urllib.request


def probs(lg: list[float]) -> list[float]:
    if len(lg) == 1:  # noul: single logit
        p = 1.0 / (1.0 + math.exp(-lg[0]))
        return [1.0 - p, p]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


def dump(url: str, workload: str, n: int, out: str) -> None:
    res = []
    with open(workload) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = {k: v for k, v in json.loads(line).items() if not k.startswith("_")}
            r["debug"] = True
            req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(r).encode(),
                                         headers={"Content-Type": "application/json"})
            body = json.loads(urllib.request.urlopen(req, timeout=120).read())
            res.append({"i": i, "logits": body["debug"]["logits"]})
    with open(out, "w") as f:
        json.dump(res, f)
    print("saved", len(res), "->", out)


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    fields = agree = 0
    max_dp = 0.0
    for x, y in zip(a, b):
        for q, la in x["logits"].items():
            pa, pb = probs(la), probs(y["logits"][q])
            fields += 1
            agree += int(max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__))
            max_dp = max(max_dp, max(abs(u - v) for u, v in zip(pa, pb)))
    print(json.dumps({"requests": min(len(a), len(b)), "fields": fields, "argmax_agree": agree, "max_abs_dp": round(max_dp, 5)}))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        dump(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
