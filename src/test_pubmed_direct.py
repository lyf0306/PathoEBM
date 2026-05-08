"""
直接测试远程 MCP search_recent_pubmed 工具。
使用 httpx 绕过本地依赖问题。
"""
import asyncio
import json
import httpx

SERVER = "117.50.173.249"
PORT = 8788  # default from code

QUERY = '(radiotherapy OR chemotherapy) AND PORTEC-3 AND endometrial cancer AND (2018:2026[dp])'
BASE = f"http://{SERVER}:{PORT}"

async def test_tool_direct():
    print("="*60)
    print(f"🔍 MCP 直连测试")
    print(f"服务器: {BASE}")
    print(f"查询: {QUERY}")
    print("="*60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Check server root
        try:
            resp = await client.get(f"{BASE}/api/list_mcps")
            print(f"\n/api/list_mcps: {resp.status_code}")
            if resp.status_code == 200:
                print(json.dumps(resp.json(), indent=2, ensure_ascii=False)[:1000])
        except Exception as e:
            print(f"  list_mcps failed: {e}")

        # 2. Try to call ncbi_mcp search_recent_pubmed via streaming HTTP
        url = f"{BASE}/ncbi_mcp/mcp/"
        print(f"\n📤 POST {url}")
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_recent_pubmed",
                "arguments": {
                    "query": QUERY,
                    "max_results": 5,
                    "retmax": 5,
                    "top_k": 5
                }
            }
        }
        try:
            resp = await client.post(url, json=payload)
            print(f"Status: {resp.status_code}")
            text = resp.text
            print(f"Response ({len(text)} chars):")
            print(text[:3000])
        except Exception as e:
            print(f"Error: {e}")

        # 3. Also try with simpler query
        print("\n" + "="*60)
        simple_q = "(PORTEC-3) AND (radiotherapy) AND (chemotherapy) AND (2018:2026[dp])"
        print(f"🔍 对比: {simple_q}")
        payload["params"]["arguments"]["query"] = simple_q
        try:
            resp = await client.post(url, json=payload)
            print(f"Status: {resp.status_code}")
            print(resp.text[:3000])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_direct())
