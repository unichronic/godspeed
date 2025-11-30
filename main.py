import subprocess
import sys
import signal
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn

# --- CONFIGURATION ---
# Paths based on exploration of 
MODEL_PATH = "./models/qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf"
DRAFT_PATH = "./models/qwen2.5-1.5b-instruct-q6_k.gguf"
BINARY_PATH = "./llama.cpp/build/bin/llama-server" 

INTERNAL_PORT = 9999
API_PORT = 5000

server_process = None
http_client = None

async def stream_generator(response):
    """Yields chunks from the backend response."""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifecycle of the llama-server subprocess and http client."""
    global server_process, http_client
    
    http_client = httpx.AsyncClient(timeout=None)
    print("🚀 Booting HeteroFlow Engine...")
    

    cmd = [
        BINARY_PATH,
        "-m", MODEL_PATH,
        "--model-draft", DRAFT_PATH,
        "--port", str(INTERNAL_PORT),
        "-ngl", "0",            # Target layers on CPU
        "-ngld", "99",          # Draft layers on RTX 2050
        "--ctx-size", "4096",   # Context window
        "--batch-size", "512",  # Batch size for verification
        "--draft-max", "12",     # Speculative lookahead
        "--n-gpu-layers", "0"   # Explicitly ensure main model is CPU only
    ]
    
    try:
        # Launch in background
        server_process = subprocess.Popen(
            cmd, 
            stdout=sys.stdout, 
            stderr=sys.stderr,
            start_new_session=True 
        )
        print(f"✅ Engine started with PID: {server_process.pid}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find binary at {BINARY_PATH}")
        print("Please compile llama.cpp and place llama-server in ./bin/")
    except Exception as e:
        print(f"❌ Failed to start engine: {e}")

    yield 

    # Cleanup on shutdown
    if http_client:
        await http_client.aclose()
        
    if server_process:
        print("🛑 Shutting down engine...")
        server_process.send_signal(signal.SIGINT)
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️ Engine did not exit gracefully, forcing kill...")
            server_process.kill()

app = FastAPI(lifespan=lifespan, title="HeteroFlow API")

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    """Proxies chat completion requests to the internal llama-server."""
    try:
        body = await request.json()
        
        req = http_client.build_request(
            "POST",
            f"http://127.0.0.1:{INTERNAL_PORT}/v1/chat/completions",
            json=body,
            timeout=None 
        )
        
        r = await http_client.send(req, stream=True)
        
        return StreamingResponse(
            stream_generator(r),
            status_code=r.status_code,
            media_type="text/event-stream"
        )
            
    except httpx.ConnectError:
        return {"error": "Engine not ready or unreachable. Is llama-server running?"}
    except Exception as e:
        return {"error": f"Proxy error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
