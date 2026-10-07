"""Generate realistic Clef SystemOne request workloads (JSONL) with calibrated token lengths.

Runs where the Clef tokenizer is available (dev pod). Each line is a full request body plus
``_tokens`` (exact encoded input length). Profiles:

  short    ~500 tok   chat-message moderation, 3 fields
  medium   ~1.5K tok  support-ticket triage, 5 fields
  long     ~4K tok    document review, 8 fields
  xlong    ~16K tok   long incident report, 4 fields (hits the 16,384 max_length)
  mixed    50% short / 35% medium / 15% long
  banking77 real BANKING77 test queries, one 77-way choice field (needs the dataset)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

SUBJECTS = ["the customer", "our team", "the billing system", "the API gateway", "the mobile app", "the vendor",
            "the support agent", "the checkout service", "the database cluster", "the account owner", "the contract",
            "the new release", "the dashboard", "the payment processor", "the shipping partner", "the security team"]
VERBS = ["reported", "noticed", "confirmed", "requested", "escalated", "rejected", "approved", "updated", "flagged",
         "investigated", "documented", "resolved", "delayed", "questioned", "expected", "acknowledged"]
OBJECTS = ["an unexpected charge on the latest invoice", "intermittent timeouts during peak hours",
           "a missing shipment for order 48213", "a request to upgrade the subscription plan",
           "elevated error rates after the deployment", "a renewal clause with a sixty day notice period",
           "duplicate notifications sent to every user", "a refund for the unused portion of the term",
           "slow page loads in the European region", "an outdated tax identifier on the account",
           "a suspicious login from an unfamiliar location", "data retention obligations for backups",
           "a failed password reset flow", "a discount that was not applied at checkout",
           "a change in the service level agreement", "a crash when uploading large attachments"]
TAILS = ["before the end of the week", "according to the logs", "after several attempts", "without prior notice",
         "as discussed in the previous call", "despite the earlier fix", "for the third time this month",
         "and asked for an update", "which affected several customers", "pending review by finance",
         "in line with the agreement", "while the incident was ongoing"]


def sentence(rng: random.Random) -> str:
    s = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}."
    return s[0].upper() + s[1:]


def text(rng: random.Random, n_words: int) -> str:
    out, count = [], 0
    while count < n_words:
        s = sentence(rng)
        out.append(s)
        count += len(s.split())
    return " ".join(out)


SCHEMAS = {
    "short": {
        "harmful": {"type": "noul", "instructions": "Does the message contain harassment, hate speech, or threats?"},
        "intent": {"type": "choice", "instructions": "What is the primary intent of the message?",
                   "criteria": {"question": "Asking for information or help", "complaint": "Expressing dissatisfaction",
                                "feedback": "Sharing an opinion or suggestion", "spam": "Unsolicited promotion or irrelevant content"}},
        "sentiment": {"type": "score", "instructions": "Rate the overall sentiment of the message.",
                      "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]},
    },
    "medium": {
        "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                       "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs, outages, performance",
                                    "account": "Login, access, profile changes", "shipping": "Deliveries and returns",
                                    "sales": "Upgrades, pricing, new purchases", "legal": "Contracts, compliance, privacy"}},
        "urgency": {"type": "score", "instructions": "How quickly does this need a response?",
                    "criteria": ["Can wait", "This week", "Today", "Immediately"]},
        "outage": {"type": "noul", "instructions": "Is a production service down or degraded?"},
        "refund": {"type": "noul", "instructions": "Is the customer requesting a refund or credit?"},
        "churn_risk": {"type": "score", "instructions": "How likely is the customer to cancel?",
                       "criteria": ["Low", "Medium", "High"]},
    },
    "long": {
        "doc_type": {"type": "choice", "instructions": "What kind of document is this?",
                     "criteria": {"msa": "Master services agreement", "nda": "Non-disclosure agreement",
                                  "sow": "Statement of work", "dpa": "Data processing addendum", "other": "Anything else"}},
        "auto_renew": {"type": "noul", "instructions": "Does the agreement renew automatically?"},
        "termination": {"type": "choice", "instructions": "Under what conditions can the customer terminate?",
                        "criteria": {"convenience": "At any time with notice", "cause": "Only for material breach",
                                     "none": "No termination right is stated"}},
        "liability_cap": {"type": "noul", "instructions": "Is there a cap on liability?"},
        "data_risk": {"type": "score", "instructions": "Rate the data protection risk.",
                      "criteria": ["Minimal", "Low", "Moderate", "High", "Severe"]},
        "governing_law": {"type": "choice", "instructions": "Which governing law applies?",
                          "criteria": {"us": "A US state", "uk": "England and Wales", "eu": "An EU member state",
                                       "unspecified": "Not specified"}},
        "needs_legal": {"type": "noul", "instructions": "Should legal review this before signature?"},
        "complexity": {"type": "score", "instructions": "How complex is the document?",
                       "criteria": ["Simple", "Standard", "Complex"]},
    },
    "xlong": {
        "root_cause": {"type": "choice", "instructions": "What was the most likely root cause of the incident?",
                       "criteria": {"deploy": "A bad deployment or config change", "capacity": "Capacity exhaustion",
                                    "dependency": "A third-party dependency failure", "security": "An attack or breach",
                                    "hardware": "Hardware or network failure"}},
        "customer_impact": {"type": "score", "instructions": "How severe was the customer impact?",
                            "criteria": ["None", "Minor", "Major", "Critical"]},
        "data_loss": {"type": "noul", "instructions": "Was any customer data lost or exposed?"},
        "follow_up": {"type": "noul", "instructions": "Are follow-up action items required?"},
    },
}


def make_state(profile: str, rng: random.Random, words: int):
    if profile == "short":
        return {"channel": rng.choice(["forum", "chat", "email", "review"]),
                "author": {"id": f"u_{rng.randint(1000, 99999)}", "account_age_days": rng.randint(1, 3000)},
                "message": text(rng, words)}
    if profile == "medium":
        return {"ticket": {"id": f"T-{rng.randint(100000, 999999)}", "subject": sentence(rng),
                           "body": text(rng, words),
                           "customer": {"plan": rng.choice(["free", "pro", "business", "enterprise"]),
                                        "mrr_usd": rng.randint(0, 20000), "region": rng.choice(["us-east", "eu-west", "apac"])},
                           "previous_tickets": rng.randint(0, 12)}}
    if profile == "long":
        n_sec = 8
        return {"document": {"title": "Agreement " + str(rng.randint(1, 999)),
                             "sections": [{"heading": f"Section {i + 1}", "text": text(rng, words // n_sec)} for i in range(n_sec)]}}
    if profile == "xlong":
        return {"incident": {"id": f"INC-{rng.randint(1000, 9999)}", "summary": sentence(rng),
                             "timeline": [f"{h:02d}:{m:02d} {sentence(rng)}" for h, m in ((rng.randint(0, 23), rng.randint(0, 59)) for _ in range(20))],
                             "notes": text(rng, words)}}
    raise ValueError(profile)


TARGETS = {"short": 500, "medium": 1500, "long": 4000, "xlong": 16000}


def calibrated(profile: str, rng: random.Random, target: int, encoder) -> dict:
    words = max(10, int(target * 0.7))
    for _ in range(6):
        req = {"model": "clef", "state": make_state(profile, random.Random(rng.random()), words), "questions": SCHEMAS[profile]}
        n = len(encoder.encode(req)[0].input_ids)
        if abs(n - target) <= max(16, target * 0.02) or n >= encoder.max_length:
            break
        words = max(10, int(words * target / n))
    req["_tokens"] = n
    req["_profile"] = profile
    return req


def banking77(n: int, seed: int):
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "mteb/banking77"
    files = [f for f in list_repo_files(repo, repo_type="dataset") if "test" in f and (f.endswith(".parquet") or f.endswith(".jsonl") or f.endswith(".jsonl.gz"))]
    path = hf_hub_download(repo, files[0], repo_type="dataset")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True)
    label_col = "label_text" if "label_text" in df.columns else "label"
    labels = sorted(df[label_col].unique().tolist())
    criteria = {lab: lab.replace("_", " ") for lab in labels}
    question = {"intent": {"type": "choice", "instructions": "Which banking intent best matches the customer's message?", "criteria": criteria}}
    df = df.sample(frac=1.0, random_state=seed).head(n)
    out = []
    for _, row in df.iterrows():
        out.append({"model": "clef", "state": row["text"], "questions": question, "_label": row[label_col], "_profile": "banking77"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/cache/workloads")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--banking-n", type=int, default=1000)
    args = ap.parse_args()
    from transformers import AutoProcessor

    from clef_engine import MODEL_PATH, CachedEncoder

    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    enc = CachedEncoder(proc.tokenizer, proc, max_length=16384)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    for profile, target in TARGETS.items():
        n = args.n if profile != "xlong" else max(60, args.n // 5)
        with open(f"{args.out}/{profile}.jsonl", "w") as f:
            for _ in range(n):
                t = int(target * (1 + rng.uniform(-args.jitter, args.jitter))) if profile != "xlong" else target
                f.write(json.dumps(calibrated(profile, rng, t, enc)) + "\n")
        print(profile, "done")
    # mixed
    pools = {p: [json.loads(line) for line in open(f"{args.out}/{p}.jsonl")] for p in ("short", "medium", "long")}
    with open(f"{args.out}/mixed.jsonl", "w") as f:
        for _ in range(args.n):
            p = rng.choices(["short", "medium", "long"], weights=[0.5, 0.35, 0.15])[0]
            f.write(json.dumps(rng.choice(pools[p])) + "\n")
    # fixed-length variants for engine/SGLang comparisons
    for profile in ("short", "medium", "long"):
        with open(f"{args.out}/{profile}_fixed.jsonl", "w") as f:
            for _ in range(args.n):
                f.write(json.dumps(calibrated(profile, rng, TARGETS[profile], enc)) + "\n")
    try:
        rows = banking77(args.banking_n, args.seed)
        with open(f"{args.out}/banking77.jsonl", "w") as f:
            for r in rows:
                r["_tokens"] = len(enc.encode(r)[0].input_ids)
                f.write(json.dumps(r) + "\n")
        print("banking77 done", len(rows), "avg tokens", sum(r["_tokens"] for r in rows) / len(rows))
    except Exception as exc:  # noqa: BLE001
        print("banking77 failed:", repr(exc))
    for name in sorted(os.listdir(args.out)):
        rows = [json.loads(line) for line in open(f"{args.out}/{name}")]
        toks = sorted(r["_tokens"] for r in rows)
        print(f"{name:22s} n={len(rows):5d} tokens p50={toks[len(toks) // 2]} min={toks[0]} max={toks[-1]} mean={sum(toks) / len(toks):.0f}")


if __name__ == "__main__":
    main()
