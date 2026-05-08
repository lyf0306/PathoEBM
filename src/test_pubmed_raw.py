"""
MCP search_recent_pubmed 诊断脚本。
在服务器上运行: python test_pubmed_raw.py
"""
import asyncio
import sys
sys.path.insert(0, "src")

from local_deep_research.connect_mcp import OrigeneMCPToolClient, mcp_servers

# 两个查询对比
QUERIES = [
    # Query 1 (成功): 带括号、无 OR 展开
    '(PORTEC-3) AND (radiotherapy) AND (chemotherapy) AND (2018:2026[dp])',
    # Query 2 (失败): 有 OR 展开
    '(radiotherapy OR chemotherapy) AND PORTEC-3 AND endometrial cancer AND (2018:2026[dp])',
    # Query 2b: 去掉 endometrial cancer
    '(radiotherapy OR chemotherapy) AND PORTEC-3 AND (2018:2026[dp])',
    # Query 2c: 只用 PORTEC-3 + 年份
    'PORTEC-3 AND (2018:2026[dp])',
]

async def main():
    print("="*60)
    print("🔍 MCP search_recent_pubmed 诊断")
    print("="*60)

    client = OrigeneMCPToolClient(mcp_servers)
    await client.initialize()
    print(f"✅ 连接成功, {len(client.mcp_tools)} 个工具\n")

    tool = client.mcp_tool_map.get("search_recent_pubmed")
    if not tool:
        print("❌ search_recent_pubmed 未找到!")
        return

    for q in QUERIES:
        print(f"\n{'='*50}")
        print(f"📝 查询: {q}")
        print(f"{'='*50}")
        try:
            result = await tool.ainvoke({
                "query": q,
                "max_results": 5,
                "retmax": 5,
                "top_k": 5,
            })
            # 打印原始返回
            print(f"类型: {type(result).__name__}")
            result_str = str(result)
            print(f"长度: {len(result_str)} 字符")
            print(f"前2000字符:\n{result_str[:2000]}")
            if len(result_str) > 2000:
                print(f"... (省略 {len(result_str)-2000} 字符)")

            # 检查是否包含 PORTEC
            if "PORTEC" in result_str:
                print("✅ 结果中包含 PORTEC!")
            else:
                print("❌ 结果中不包含 PORTEC")
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
