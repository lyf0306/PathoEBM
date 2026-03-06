"""
测试MCP服务器连通性

运行: python test_mcp_connection.py
"""
import asyncio
from local_deep_research.connect_mcp import OrigeneMCPToolClient, mcp_servers


async def test_connection():
    """测试MCP服务器连接"""
    print("="*60)
    print("🔍 测试MCP服务器连接")
    print("="*60)
    
    # 1. 显示配置的服务器
    print("\n📋 配置的MCP服务器:")
    for pkg, config in mcp_servers.items():
        print(f"   - {pkg}: {config['url']}")
    
    # 2. 尝试连接
    print("\n🔌 正在连接MCP服务器...")
    try:
        client = OrigeneMCPToolClient(mcp_servers)
        await client.initialize()
        
        print(f"\n✅ 连接成功! 共加载 {len(client.mcp_tools)} 个工具")
        
        # 3. 显示可用工具
        print("\n📦 可用工具列表:")
        for tool_name, source in sorted(client.tool2source.items()):
            print(f"   [{source:20s}] {tool_name}")
        
        # 4. 测试临床审计所需工具
        print("\n🏥 临床审计关键工具检查:")
        required_tools = {
            "get_studies": "临床试验检索",
            "get_indications_by_drug_name": "FDA药物适应症",
            "get_gene_metadata_by_gene_name": "基因信息查询",
            "clinvar_query_variant_significance": "变异致病性判断",
            "get_general_info_by_disease_name": "疾病信息查询",
        }
        
        for tool, desc in required_tools.items():
            status = "✅" if tool in client.mcp_tool_map else "❌"
            print(f"   {status} {tool:40s} - {desc}")
        
        print("\n" + "="*60)
        print("✅ 测试完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n💡 故障排查:")
        print("   1. 确认MCP服务器已启动: uv run -m deploy.web")
        print("   2. 检查端口8788是否被占用")
        print("   3. 验证.secrets.toml中的mcp.server_url配置")
        raise


if __name__ == "__main__":
    asyncio.run(test_connection())