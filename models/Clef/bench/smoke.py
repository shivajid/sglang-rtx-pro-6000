"""One request against a Clef server; prints status, latency and the response.

python smoke.py [url]      (default http://clef-server:8000)
"""

import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://clef-server:8000"
REQUEST = {
    "model": "clef",
    "state": "Hi, my card payment was declined twice at the grocery store today even though I have money in my account.",
    "questions": {
        "topic": {
            "type": "choice",
            "instructions": "What is the customer asking about?",
            "criteria": {"card_declined": "A card payment was declined", "transfer": "Money transfers", "other": "Anything else"},
        },
        "urgent": {"type": "noul", "instructions": "Does the customer need help right now?"},
        "frustration": {"type": "score", "criteria": ["calm", "annoyed", "angry"]},
    },
}

t = time.time()
req = urllib.request.Request(URL + "/v1/systemone", data=json.dumps(REQUEST).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    body = r.read().decode()
    print("status", r.status, "latency_ms", round((time.time() - t) * 1000, 1))
print(json.dumps(json.loads(body), indent=1)[:1500])
