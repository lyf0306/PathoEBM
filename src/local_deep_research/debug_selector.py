import asyncio
import os
import sys

# 1. 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取父目录 (即 src 目录)
src_dir = os.path.dirname(current_dir)
# 3. 将 src 目录添加到系统路径，这样我们就能以 local_deep_research.xxx 的方式导入
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 4. 使用完整的包路径导入 (解决 relative import 问题)
# 注意：不要使用 "from .config" 或 "from connect_mcp"
try:
    from local_deep_research.connect_mcp import OrigeneMCPToolClient, mcp_servers
    from local_deep_research.tool_selector import ToolSelector
    from local_deep_research.config import get_deepseek_v3, get_deepseek_r1  # 根据您的 config.py 实际内容调整
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在 src 目录下运行，并且不要修改 debug_selector.py 中的导入路径。")
    sys.exit(1)

async def test_selector():
    print("\n🧠 Testing Tool Selector Logic...")
    
    # 1. 初始化 Mock 客户端
    # 注意：确保 connect_mcp.py 中的 mcp_servers 配置正确
    try:
        client = OrigeneMCPToolClient(mcp_servers, None)
        await client.initialize()
    except Exception as e:
        print(f"❌ MCP Client 初始化失败: {e}")
        return

    # 2. 初始化 Selector
    # 这里使用您 config.py 中定义的模型获取函数
    # 如果您使用的是 GPT-4，请替换为 get_gpt4_1()
    try:
        llm = get_deepseek_v3() 
        reasoning_llm = get_deepseek_r1()
    except Exception as e:
        print(f"❌ 模型加载失败 (检查 config.py): {e}")
        return
    
    # 在纯 API 模式下，后面几个参数传 None 即可
    selector = ToolSelector(
        llm_light=llm,
        llm_reasoning=reasoning_llm,
        mcp_tool_client=client,
        tool_info_data=None, 
        embedding_api_key=None, 
        embedding_cache=None, 
        available_tools=None
    )
    
    # 3. 测试查询
    test_query = "Check latest 2024 survival data for Pembrolizumab in Endometrial Cancer from literature."
    print(f"   Query: {test_query}")
    
    tools = await selector.run(test_query)
    
    print(f"   Selected Tools: {tools}")
    
    # 验证逻辑
    if tools and isinstance(tools, list) and len(tools) > 0:
        tool_name = tools[0].get('tool_name')
        if tool_name == 'search_recent_pubmed':
            print("✅ Selector Logic PASS: Correctly chose PubMed tool.")
        elif tool_name in ['get_studies', 'get_indications_by_drug_name']:
            print(f"✅ Selector Logic PASS: Chose valid API tool ({tool_name}).")
        else:
            print(f"⚠️ Selector Logic WARNING: Chose unexpected tool: {tool_name}")
    else:
        print("❌ Selector Logic FAIL: No tools selected.")

if __name__ == "__main__":
    asyncio.run(test_selector())