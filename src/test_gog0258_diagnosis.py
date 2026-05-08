"""
诊断脚本：对比 MCP search_recent_pubmed 与 PubMed 网站的返回差异。

问题：在 PubMed 网站搜索 "GOG-0258 AND (PFS OR OS) AND endometrial cancer AND 2018:2026[dp]"
能搜到 PMID 39854806 (GOG-0258 分子分型辅助分析)，但 Agent 通过 MCP 搜同一检索词却说
"缺乏具体的生存数据"。

此脚本测试：
  1. MCP 返回了哪几篇文献（PMID + 标题）
  2. max_results=3 vs 5 的差异
  3. PMID 39854806 是否在结果中
  4. 每篇返回的原始文本长度（判断是否被截断导致数据丢失）
"""
import asyncio
import json
import sys
sys.path.insert(0, r"C:\Users\Lenovo\Desktop\PathoEBM-main\src")

from local_deep_research.connect_mcp import OrigeneMCPToolClient, mcp_servers

# ── 待诊断的检索词 ──
QUERY = 'GOG-0258 AND (PFS OR OS) AND endometrial cancer AND 2018:2026[dp]'

# PMID 39854806 是用户在 PubMed 网站搜到的目标文献
TARGET_PMID = "39854806"


async def diagnose():
    print("=" * 70)
    print("🔍 GOG-0258 MCP 检索诊断")
    print(f"检索词: {QUERY}")
    print(f"目标文献: PMID {TARGET_PMID}")
    print("=" * 70)

    # 1. 连接 MCP
    print("\n🔌 连接 MCP 服务器...")
    client = OrigeneMCPToolClient(mcp_servers)
    await client.initialize()
    print(f"✅ 已连接, {len(client.mcp_tools)} 个工具可用")

    # 2. 获取 search_recent_pubmed 工具
    tool = client.mcp_tool_map.get("search_recent_pubmed")
    if not tool:
        print("❌ search_recent_pubmed 未找到!")
        print(f"可用工具: {list(client.mcp_tool_map.keys())}")
        return

    # 3. 分别测试 max_results=3 和 max_results=5
    for max_r in [3, 5]:
        print(f"\n{'=' * 70}")
        print(f"📤 max_results={max_r}")
        print(f"{'=' * 70}")

        tool_input = {
            "query": QUERY,
            "max_results": max_r,
            "retmax": max_r,
            "top_k": max_r,
        }
        try:
            result = await tool.ainvoke(tool_input)
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            continue

        # 解析结果
        print(f"\n📥 原始返回类型: {type(result).__name__}")

        # MCP 返回格式通常是 list[dict] 或 str
        articles = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "content" in item:
                    content = item["content"]
                    try:
                        parsed = eval(content)  # MCP 返回序列化的 list
                        if isinstance(parsed, list):
                            articles.extend(parsed)
                        else:
                            articles.append(item)
                    except Exception:
                        articles.append(item)
                else:
                    articles.append(item)
        elif isinstance(result, str):
            try:
                parsed = eval(result)
                if isinstance(parsed, list):
                    articles = parsed
            except Exception:
                articles = [{"raw": result}]

        print(f"解析出 {len(articles)} 篇文献\n")

        found_target = False
        for i, art in enumerate(articles):
            text = ""
            if isinstance(art, dict) and "text" in art:
                text = art["text"]
            elif isinstance(art, str):
                text = art

            # 提取 PMID
            import re
            pmid_matches = re.findall(r'PMID[:\s]*(\d{7,9})', text, re.IGNORECASE)
            pmids = pmid_matches or ["未找到"]

            # 提取标题
            title_match = re.search(r'Title:\s*([^\n]+)', text, re.IGNORECASE)
            title = title_match.group(1).strip()[:120] if title_match else "未提取到标题"

            # 检查是否命中目标
            is_target = TARGET_PMID in pmids

            print(f"--- 文献 {i+1} ---")
            print(f"  PMID(s): {', '.join(pmids)}")
            print(f"  标题: {title}")
            print(f"  文本长度: {len(text)} 字符")
            print(f"  含关键词 'ancillary': {'ancillary' in text.lower()}")
            print(f"  含 'p53abn': {'p53abn' in text.lower()}")
            print(f"  含 'hazard ratio' 或 'HR': {bool(re.search(r'hazard\s*ratio|HR\s*=', text, re.IGNORECASE))}")
            if is_target:
                print(f"  🎯🎯🎯 命中目标文献! PMID {TARGET_PMID} 在此结果中!")
                found_target = True

            # 显示摘要前 500 字符
            abstract_match = re.search(r'Abstract:\s*(.{0,500})', text, re.IGNORECASE | re.DOTALL)
            if abstract_match:
                print(f"  摘要前500字: {abstract_match.group(1).strip()[:500]}")

        if not found_target:
            print(f"\n❌ PMID {TARGET_PMID} 不在 max_results={max_r} 的返回结果中!")

    # 4. 对比：尝试用更精准的检索词
    print(f"\n{'=' * 70}")
    print(f"🔬 对比测试：精准检索词")
    print(f"{'=' * 70}")

    precise_queries = [
        # 去掉 (PFS OR OS) 的限制，看看是不是这些词排除了该文献
        'GOG-0258 AND endometrial cancer AND 2018:2026[dp]',
        # 加上 molecular 定位分子分型
        'GOG-0258 AND (molecular OR p53 OR MMR) AND endometrial cancer AND 2018:2026[dp]',
        # 直接搜 PMID
        '39854806[uid]',
    ]

    for pq in precise_queries:
        print(f"\n📤 检索: {pq}")
        try:
            result = await tool.ainvoke({
                "query": pq,
                "max_results": 3,
                "retmax": 3,
                "top_k": 3,
            })
            text = str(result)[:1500]
            # 提取 PMID
            pmids_found = re.findall(r'PMID[:\s]*(\d{7,9})', text, re.IGNORECASE)
            has_target = TARGET_PMID in pmids_found
            status = "🎯 命中!" if has_target else "❌ 未命中"
            print(f"  {status}  找到 PMID: {pmids_found}")
            if has_target:
                # 提取标题确认
                title_m = re.search(r'Title:\s*([^\n]+)', text, re.IGNORECASE)
                if title_m:
                    print(f"  标题: {title_m.group(1).strip()[:150]}")
        except Exception as e:
            print(f"  ❌ 调用失败: {e}")

    print(f"\n{'=' * 70}")
    print("✅ 诊断完成")


if __name__ == "__main__":
    asyncio.run(diagnose())
