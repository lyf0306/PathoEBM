"""
直接测试 MCP search_recent_pubmed 工具，看原始返回结果。
"""
import asyncio
import json
import sys
sys.path.insert(0, r"C:\Users\Lenovo\Desktop\PathoEBM-main\src")

from local_deep_research.connect_mcp import OrigeneMCPToolClient, mcp_servers

QUERY = '(radiotherapy OR chemotherapy) AND PORTEC-3 AND endometrial cancer AND (2018:2026[dp])'

async def test_pubmed_query():
    print("="*60)
    print(f"🔍 测试 search_recent_pubmed")
    print(f"查询: {QUERY}")
    print("="*60)

    # 1. Connect
    print("\n🔌 Connecting to MCP servers...")
    client = OrigeneMCPToolClient(mcp_servers)
    await client.initialize()
    print(f"✅ Connected, {len(client.mcp_tools)} tools loaded")

    # 2. Find search_recent_pubmed tool
    tool = client.mcp_tool_map.get("search_recent_pubmed")
    if not tool:
        print("❌ search_recent_pubmed not found in tool map!")
        print(f"Available tools: {list(client.mcp_tool_map.keys())}")
        return
    print(f"✅ Found tool: {tool.name}")

    # 3. Call it
    tool_input = {
        "query": QUERY,
        "max_results": 5,
        "retmax": 5,
        "top_k": 5,
    }
    print(f"\n📤 Calling tool with input: {tool_input}")
    try:
        result = await tool.ainvoke(tool_input)
        print(f"\n📥 Raw result type: {type(result)}")
        print(f"\n📥 Raw result (str):")
        print(str(result)[:3000])

        # If it's a list, dump each element
        if isinstance(result, list):
            print(f"\n📥 Result is a list with {len(result)} elements")
            for i, item in enumerate(result):
                print(f"\n--- Element {i+1} ---")
                if isinstance(item, dict):
                    for k, v in item.items():
                        vs = str(v)[:500]
                        print(f"  {k}: {vs}")
                else:
                    print(f"  {str(item)[:500]}")
        elif isinstance(result, dict):
            print(f"\n📥 Result is a dict with keys: {list(result.keys())}")
            for k, v in result.items():
                vs = str(v)[:500]
                print(f"  {k}: {vs}")
    except Exception as e:
        print(f"\n❌ Error calling tool: {e}")
        import traceback
        traceback.print_exc()

    # 4. Also test with a simpler query for comparison
    print("\n" + "="*60)
    print("🔍 对比测试: (PORTEC-3) AND (radiotherapy) AND (chemotherapy) AND (2018:2026[dp])")
    print("="*60)
    simple_query = "(PORTEC-3) AND (radiotherapy) AND (chemotherapy) AND (2018:2026[dp])"
    simple_input = {
        "query": simple_query,
        "max_results": 5,
        "retmax": 5,
        "top_k": 5,
    }
    try:
        result2 = await tool.ainvoke(simple_input)
        print(f"\n📥 Raw result (str):")
        print(str(result2)[:3000])
        if isinstance(result2, list):
            print(f"\n📥 Result is a list with {len(result2)} elements")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_pubmed_query())
