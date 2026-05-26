from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from .config import mcp_url

MCP_SERVICE_URL = mcp_url 

# [修改点]：精简工具列表，移除导致 404 的 search_mcp 和无关工具
tool_packages = [
    # --- 核心临床审计工具 (必须保留) ---
   "clinicaltrials_mcp",  # ClinicalTrials.gov
    "fda_drug_mcp",        # FDA
    "ncbi_mcp",            # PubMed + Gene
]

mcp_servers = {
    package: {
        "transport": "streamable_http",
        "url": f"{MCP_SERVICE_URL}/{package}/mcp/",
    }
    for package in tool_packages
}

class OrigeneMCPToolClient:
    def __init__(self, mcp_servers: dict[str, Any], specified_tools: list = None):
        self.mcp_servers = mcp_servers
        self.mcp_tools = None
        self.mcp_tool_map = {}
        self.available_tools = specified_tools

    async def initialize(self):
        """Initialize async components"""
        client = MultiServerMCPClient(self.mcp_servers)

        self.tool2source = {}
        # 建立工具到来源包的映射
        for pkg_name in self.mcp_servers.keys():
            # 使用 try-except 包裹，防止单个服务挂掉影响整体
            try:
                async with client.session(pkg_name) as session:
                    # 这里的 session 只是为了获取工具列表的元数据
                    pass
            except Exception as e:
                print(f"⚠️ Warning: Could not pre-fetch metadata for {pkg_name}: {e}")

        # 获取所有工具
        try:
            self.mcp_tools = await client.get_tools()
        except Exception as e:
            print(f"❌ Critical: Failed to get tools from MCP client: {e}")
            raise e

        # 更新 tool2source 映射 (基于实际获取到的工具)
        # 注意：这里简化了逻辑，因为 MultiServerMCPClient 会自动聚合
        for tool in self.mcp_tools:
            # 简单的反向推导或默认归类
            self.tool2source[tool.name] = "unknown_mcp"

        if self.available_tools:
            self.mcp_tools = [
                tool for tool in self.mcp_tools if tool.name in self.available_tools
            ]
        self.mcp_tool_map = {tool.name: tool for tool in self.mcp_tools}
        print(f"✅ MCP server connected! Found {len(self.mcp_tools)} tools")