import requests
import time
import json
import argparse
import os
import difflib

# Configuration
API_URL = "http://localhost:5000/v1/chat/completions"
PROMPT = "write a program to python print pyramid pattern of height 5"
BASELINE_FILE = "baseline_results.json"

def run_benchmark(mode):
    label = "Baseline" if mode == "baseline" else "HeteroFlow"
    print(f"\n--- Benchmarking: {label} ---")
    print(f"Prompt: {PROMPT[:50]}...")
    
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "temperature": 0.1  # Low temp for consistent results
    }

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response_text = ""
    
    try:
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    # Mark time for the very first token (Latency)
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    try:
                        chunk_json = json.loads(decoded[6:])
                        if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                            delta = chunk_json["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response_text += content
                                token_count += 1
                                print(".", end="", flush=True)
                    except json.JSONDecodeError:
                        pass

    except Exception as e:
        print(f"\nError: {e}")
        return

    end_time = time.time()
    print() # Newline after dots
    
    # Calculations
    total_time = end_time - start_time
    # Latency: Time from sending request to seeing first word
    latency = (first_token_time - start_time) if first_token_time else 0
    # Generation Time: Time spent generating tokens (excluding latency)
    gen_time = end_time - first_token_time if first_token_time else total_time
    tps = token_count / gen_time if gen_time > 0 else 0

    results = {
        "tps": tps,
        "latency": latency,
        "total_tokens": token_count,
        "text": full_response_text
    }

    print(f"\n\n📊 RESULTS for {label}:")
    print(f"-----------------------------")
    print(f"✅ Total Tokens:    {token_count}")
    print(f"⏱️  Latency (TTFT):  {latency:.4f} seconds")
    print(f"🚀 Speed (TPS):     {tps:.2f} tokens/sec")
    print(f"-----------------------------")

    if mode == "baseline":
        with open(BASELINE_FILE, "w") as f:
            json.dump(results, f)
        print(f"💾 Baseline results saved to {BASELINE_FILE}")
        print("Now restart main.py with Draft Model ENABLED and run with --heteroflow")

    elif mode == "heteroflow":
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE, "r") as f:
                baseline = json.load(f)
            
            baseline_tps = baseline.get("tps", 0)
            if baseline_tps > 0:
                speedup = tps / baseline_tps
                print(f"⚡ Speedup vs Baseline: {speedup:.2f}x")
            
            # Accuracy Check
            similarity = difflib.SequenceMatcher(None, baseline.get("text", ""), full_response_text).ratio()
            print(f"🎯 Accuracy (Similarity): {similarity*100:.1f}%")
            
            if similarity < 0.95:
                print("⚠️  Warning: Output differs significantly from baseline!")
        else:
            print("⚠️  No baseline results found. Run with --baseline first to calculate speedup.")

        # Prompt for Acceptance Rate
        print("\n🔎 Check your main.py terminal for 'draft:' acceptance rate.")
        ar_input = input("Enter Acceptance Rate % (e.g. 62.4) or press Enter to skip: ")
        if ar_input.strip():
            print(f"📝 Recorded Acceptance Rate: {ar_input}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark HeteroFlow vs Baseline")
    parser.add_argument("--baseline", action="store_true", help="Run as Baseline (Test A)")
    parser.add_argument("--heteroflow", action="store_true", help="Run as HeteroFlow (Test B)")
    
    args = parser.parse_args()
    
    if args.baseline:
        run_benchmark("baseline")
    elif args.heteroflow:
        run_benchmark("heteroflow")
    else:
        print("Please specify --baseline or --heteroflow")
