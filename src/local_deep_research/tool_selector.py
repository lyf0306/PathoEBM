import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 确保保留原有导入，并增加工具函数导入
from .tool_embedding_retriever import ToolEmbeddingRetriever
from .search_system_support import KGNetwork, safe_json_from_text  # ✅ 导入更稳健的JSON解析器
from .utilties.search_utilities import remove_think_tags           # ✅ 导入思考标签清理工具

logger = logging.getLogger(__name__)

class GeneralToolSelector:
    """
    Selects general tools based on LLM reasoning.
    Robustly handles DeepSeek-R1 <think> tags.
    """
    
    # 权威工具白名单 (保持不变)
    GENERAL_TOOLS_NAME = [
        "get_studies", "get_study",                                  # ClinicalTrials
        "get_indications_by_drug_name", "get_warnings_by_drug_name", # FDA
        "get_contraindications_by_drug_name", "get_clinical_pharmacology_by_drug_name", 
        "get_dosage_and_storage_information_by_drug_name",
        "get_publications_by_disease_efoId", "get_publications_by_drug_chemblId", # OpenTargets
        "get_disease_id_description_by_name",
        "get_gene_metadata_by_gene_name", "clinvar_query_variant_significance",    # NCBI
        "search_recent_pubmed" # ✅ 确保包含 PubMed
    ]

    def __init__(self, llm: ChatOpenAI, llm_reasoning: ChatOpenAI, mcp_tool_client):
        self.llm = llm
        self.llm_reasoning = llm_reasoning
        self.mcp_tool_client = mcp_tool_client
        
        # 过滤可用工具
        self.available_tools = [
            tool for tool in self.mcp_tool_client.mcp_tools 
            if tool.name in self.GENERAL_TOOLS_NAME or tool.name == "search_recent_pubmed"
        ]
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are a clinical research assistant. Select authoritative tools for the query.

Query: "{query}"

Available Tools:
{tools_desc}

Instructions:
1. Analyze the query.
2. Select tools providing official evidence (FDA, NIH, PubMed).
3. Output strictly valid JSON.

Output Format:
{{
    "tool_calls": [
        {{
            "tool_name": "exact_tool_name_from_list",
            "tool_input": {{ "arg_name": "value" }}
        }}
    ]
}}
"""
        )

    async def run(self, query: str) -> List[Dict]:
        if not self.available_tools:
            logger.warning("No authoritative tools available.")
            return []

        tools_desc = "\n".join([
            f"- {t.name}: {t.description[:150]}..." 
            for t in self.available_tools
        ])

        chain = self.prompt | self.llm_reasoning
        
        try:
            response = await chain.ainvoke({"query": query, "tools_desc": tools_desc})
            
            # ✅ [关键修复 1] 清理 DeepSeek 的 <think> 标签
            content = remove_think_tags(response.content)
            
            # ✅ [关键修复 2] 使用更强的正则提取 JSON，防止 Markdown 代码块干扰
            result = safe_json_from_text(content)
            
            if not result:
                # 降级尝试：如果不含 JSON 结构，可能是空或纯文本
                logger.warning(f"Failed to parse JSON from tool selector response: {content[:100]}...")
                return []
            
            tool_calls = result.get("tool_calls", [])
            logger.info(f"General selector chose {len(tool_calls)} tools for query: {query[:50]}...")
            return tool_calls
            
        except Exception as e:
            logger.error(f"General tool selection failed: {e}")
            return []


class ExpertToolSelector:
    """
    Expert Selector (kept for compatibility structure)
    """
    def __init__(
        self,
        llm: ChatOpenAI,
        mcp_tool_client,
        tool_embedding_retriever: ToolEmbeddingRetriever,
        kg_network: KGNetwork,
    ):
        self.llm = llm
        self.mcp_tool_client = mcp_tool_client
        self.tool_embedding_retriever = tool_embedding_retriever
        self.kg_network = kg_network

    async def run(self, query: str) -> List[Dict]:
        return []

    async def extract_entity(self, query: str) -> Dict:
        return {}


class ToolSelector:
    """
    Main Entry Point
    """
    def __init__(
        self,
        llm_light: ChatOpenAI,
        llm_reasoning: ChatOpenAI,
        mcp_tool_client,
        tool_info_data: str = None,
        embedding_api_key: str = None,
        embedding_cache: str = None,
        available_tools: list = None,
    ):
        self.llm_light = llm_light
        self.llm_reasoning = llm_reasoning
        self.mcp_tool_client = mcp_tool_client
        
        # 即使在 Audit 模式下，也初始化一个空的/占位的 expert selector 以防调用出错
        self.expert_tool_selector = None

        self.general_tool_selector = GeneralToolSelector(
            self.llm_light, self.llm_reasoning, self.mcp_tool_client
        )

    async def run(self, query: str, research_mode: str = "general") -> List[Dict]:
        """
        Select tools for the query.
        """
        # 在纯 API 模式或 Audit 模式下，只使用 General Selector
        return await self.general_tool_selector.run(query)