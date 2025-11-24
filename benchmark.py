import requests
import time
import json

# Configuration
API_URL = "http://localhost:5000/v1/chat/completions"
PROMPT = "write a program to python print pyramid pattern of height 5"

def run_benchmark(label):
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
                    
                    token_count += 1
                    # Optional: Print dots to show progress
                    print(".", end="", flush=True)

    except Exception as e:
        print(f"\nError: {e}")
        return

    end_time = time.time()
    
    # Calculations
    total_time = end_time - start_time
    # Latency: Time from sending request to seeing first word
    latency = (first_token_time - start_time) if first_token_time else 0
    # Generation Time: Time spent generating tokens (excluding latency)
    gen_time = end_time - first_token_time if first_token_time else total_time
    tps = token_count / gen_time if gen_time > 0 else 0

    print(f"\n\n📊 RESULTS for {label}:")
    print(f"-----------------------------")
    print(f"✅ Total Tokens:    {token_count}")
    print(f"⏱️  Latency (TTFT):  {latency:.4f} seconds")
    print(f"🚀 Speed (TPS):     {tps:.2f} tokens/sec")
    print(f"-----------------------------")

if __name__ == "__main__":
    # Give the engine a moment to wake up if needed
    run_benchmark("HeteroFlow System")
