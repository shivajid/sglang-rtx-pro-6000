#!/usr/bin/env python3
"""
Test script for DeepSeek-Recipe Gateway on GKE (via port-forward).
Uses built-in Python standard libraries (no pip/external packages required).
"""

import json
import sys
import urllib.error
import urllib.request

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"

def run_test():
    print(f"Connecting to DeepSeek-Recipe Gateway at: {GATEWAY_URL} ...\n")
    payload = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [
            {"role": "user", "content": "Explain quantum entanglement in 2 sentences."}
        ],
        "stream": True,
    }

    req = urllib.request.Request(
        GATEWAY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    tokens_received = 0
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # Handle both standard content and reasoning content from deepseek-recipe
                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")
                    if reasoning:
                        print(reasoning, end="", flush=True)
                        tokens_received += 1
                    if content:
                        print(content, end="", flush=True)
                        tokens_received += 1
                except json.JSONDecodeError:
                    continue
        if tokens_received > 0:
            print(f"\n\n[SUCCESS] Stream completed ({tokens_received} token chunks received).")
        else:
            print("\n[WARNING] Stream completed without receiving any content tokens.")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"\n[HTTP Error {e.code}]: {err}")
    except urllib.error.URLError as e:
        print(f"\n[Connection Error]: {e.reason}")
        print("Hint: Make sure 'kubectl port-forward svc/deepseek-recipe-gateway-svc 8000:8000' is running.")

if __name__ == "__main__":
    run_test()
