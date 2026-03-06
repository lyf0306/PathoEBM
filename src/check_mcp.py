import httpx
import asyncio

async def check_endpoints():
    base = "http://127.0.0.1:8788"
    package = "chembl_mcp"
    
    # List of potential paths to check
    paths = [
        f"/{package}/sse",       # Standard SSE endpoint (GET)
        f"/{package}/messages",  # Standard Messages endpoint (POST)
        f"/{package}/",          # Stateless root endpoint (POST)
        f"/{package}",           # Stateless root without slash
        f"/mcp/{package}/sse",   # Alternative mounting
        f"/{package}/mcp/sse",       # ✅ 这是真正的 SSE 端点
        f"/{package}/mcp/messages",  # ✅ 这是真正的 POST 端点
        "/api/list_mcps"         # Server meta-endpoint
    ]

    print(f"🔍 Probing MCP Server at {base} for '{package}'...")
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        # 1. Check list_mcps to see what the server claims
        try:
            resp = await client.get(f"{base}/api/list_mcps")
            print(f"\n[GET] /api/list_mcps: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    Server claims URL for {package}: {data.get(package, {}).get('url', 'Not found')}")
        except Exception as e:
            print(f"    Failed to query list_mcps: {e}")

        print("\nChecking specific endpoints:")
        # 2. Probe paths
        for path in paths:
            url = f"{base}{path}"
            
            # Try GET (for SSE)
            try:
                resp = await client.get(url)
                if resp.status_code != 404:
                    print(f"✅ [GET]  {path:<25} -> {resp.status_code} (Possible SSE endpoint)")
                else:
                    pass # print(f"❌ [GET]  {path:<25} -> 404")
            except: pass

            # Try POST (for Stateless/Messages)
            try:
                # Sending an empty JSON-RPC like body to see if it responds
                resp = await client.post(url, json={"jsonrpc": "2.0", "method": "ping", "id": 1})
                if resp.status_code != 404:
                    print(f"✅ [POST] {path:<25} -> {resp.status_code} (Possible Action endpoint)")
                    if resp.status_code == 200:
                        print(f"       Response: {resp.text[:100]}...")
                else:
                    pass
            except: pass

if __name__ == "__main__":
    asyncio.run(check_endpoints())