import json
from typing import List
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .utils import extract_and_convert_dict, exact_match_entity_type

# === 新增：引入 OpenAI 官方客户端（用于兼容各厂商的本地 Embedding API 服务） ===
from openai import OpenAI

class APIQwenEmbedding:
    """
    通过 HTTP API 调用 Embedding 服务，彻底解耦本地显卡与 torch 依赖。
    你可以使用 vLLM、Xinference 或沐曦官方支持的推理框架在 8002 端口启动模型。
    """
    def __init__(self, base_url="http://localhost:8002/v1", api_key="EMPTY", model_name="QwenEmbedding"):
        print(f"🔄 [Embedding] Connecting to API: {base_url}")
        try:
            # 初始化同步 HTTP 客户端
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model_name = model_name
            print("✅ [Embedding] API Client initialized successfully.")
        except Exception as e:
            print(f"❌ [Embedding] Error initializing API client: {e}")
            self.client = None
    
    def embed_query(self, query: str) -> np.ndarray:
        if self.client is None:
            return np.zeros(2560)
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=query
            )
            # OpenAI API 返回的通常已经是 normalize 过的向量
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"❌ [Embedding] Error fetching embedding for query: {e}")
            return np.zeros(2560)
    
    def embed_documents(self, query_list: List[str]) -> List[np.ndarray]:
        if self.client is None or not query_list:
            return [np.zeros(2560) for _ in query_list]
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=query_list
            )
            return [np.array(data.embedding) for data in response.data]
        except Exception as e:
            print(f"❌ [Embedding] Error fetching embedding for documents: {e}")
            return [np.zeros(2560) for _ in query_list]


class ToolEmbeddingRetriever:
    
    def __init__(self, llm, mcp_tool_client, embedding_api_key: str, embedding_cache: str, available_tools: list = None):
        self.llm = llm
        self.mcp_tool_client = mcp_tool_client
        self.mcp_tool_map = mcp_tool_client.mcp_tool_map
        
        # === 修改：使用 API 客户端替代本地本地大模型加载 ===
        # 这里你可以将 base_url 放到 config 中统一管理，目前默认指向 8002 端口
        self.embedding_model = APIQwenEmbedding(
            base_url="http://localhost:8002/v1", 
            api_key=embedding_api_key if embedding_api_key else "EMPTY"
        )
        
        self._load_tool_embedding_cache(embedding_cache)
        self.tool_name_list = list(mcp_tool_client.mcp_tool_map.keys())
        if available_tools:
            self.tool_name_list = [tool_name for tool_name in self.tool_name_list if tool_name in available_tools]
        
        # 确保缓存不是空的，防止报错
        if self.tool_embedding_cache:
            self.all_tool_embedding = np.array(list(self.tool_embedding_cache.values()))
        else:
            self.all_tool_embedding = np.zeros((0, 2560)) # 占位符
        
    def _load_tool_embedding_cache(self, embedding_cache_path: str):
        # Create directory first if it doesn't exist
        if not os.path.exists(os.path.dirname(embedding_cache_path)):
            os.makedirs(os.path.dirname(embedding_cache_path))
            
        # Load existing cache if available
        if os.path.exists(embedding_cache_path):
            try:
                with open(embedding_cache_path, 'rb') as f:
                    data = pkl.load(f)
                    
                # 兼容性处理：如果 pkl 是新的格式 (dict 包含 'tool_embeddings') 或旧格式 (直接是 dict)
                if isinstance(data, dict) and "tool_embeddings" in data and "tool_names" in data:
                    self.tool_embedding_cache = {}
                    names = data["tool_names"]
                    embeddings = data["tool_embeddings"]
                    for name, emb in zip(names, embeddings):
                        self.tool_embedding_cache[name] = emb
                    print(f"✅ Loaded {len(self.tool_embedding_cache)} tools from new PKL format.")
                else:
                    self.tool_embedding_cache = data
                    print(f"✅ Loaded {len(self.tool_embedding_cache)} tools from legacy PKL format.")
                    
            except Exception as e:
                print(f"❌ Error loading PKL: {e}")
                self.tool_embedding_cache = {}
        else:
            self.tool_embedding_cache = {}
            
        cached_embedding_num = len(self.tool_embedding_cache)
        
        if not hasattr(self, 'mcp_tool_map'):
            print("Warning: mcp_tool_map not initialized yet")
            return
            
        for tool_name, tool in tqdm(self.mcp_tool_map.items()):
            if tool_name in self.tool_embedding_cache:
                continue
                
            if not hasattr(tool, 'description'):
                print(f"Warning: tool {tool_name} has no description")
                continue
                
            tool_func_desc = tool.description.split('Args')[0]
            embedding = self.embedding_model.embed_query(tool_func_desc)
            
            # Check if embedding is valid (not a zero vector fallback)
            if embedding is not None and np.any(embedding):
                self.tool_embedding_cache[tool_name] = embedding
            else:
                print(f"Warning: Failed to generate valid embedding for {tool_name}")
        
        if len(self.tool_embedding_cache) > cached_embedding_num:
            try:
                if not os.path.exists(os.path.dirname(embedding_cache_path)):
                    os.makedirs(os.path.dirname(embedding_cache_path))
                with open(embedding_cache_path, 'wb') as f:
                    pkl.dump(self.tool_embedding_cache, f)
                print(f"Tool embedding cache saved to {embedding_cache_path}")
            except Exception as e:
                print(f"Error saving embedding cache: {e}")
    
    def retrieve_tools(self, query: str, top_k: int = 5, explain_item: bool = False):
        if explain_item:
            query = self.explain_item(query)
            
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        if np.isnan(query_embedding).any() or (len(self.all_tool_embedding) > 0 and np.isnan(self.all_tool_embedding).any()):
            print("Warning: Embedding contains NaN.")
            return [], []
            
        if len(self.all_tool_embedding) == 0:
            return [], []

        similarities = cosine_similarity(query_embedding, self.all_tool_embedding)[0]
        _tools = list(zip(self.tool_name_list, similarities))
            
        sorted_tools = sorted(_tools, key=lambda x: x[1], reverse=True)
        
        k = min(top_k, len(sorted_tools))
        if k == 0:
            return [], []
            
        top_k_tools, top_k_scores = zip(*sorted_tools[:k])
        
        return list(top_k_tools), list(top_k_scores)
    
    def explain_item(self, item: str):
        prompt = f"""
        You are an experienced disease biologist, please briefly explain what this is: {item}
        Note: 1. Please summarize in one paragraph; 2. Do not use bullet points; 3. Keep the answer under 100 words.
        """
        try:
            result = self.llm.invoke(prompt)
            return result.content
        except:
            return item
    
    def batch_explain_item(self, item_list: list):
        prompt = f"""
        ## Task Description
        You are an experienced disease biologist, please briefly explain what these are: {item_list}
        ## Notes
        1. Please summarize in one paragraph; 
        2. Do not use bullet points; 
        3. Keep the answer under 100 words.
        ## Output Format
        Please provide your analysis in the following JSON format:
        ```json
        {{
            "item1": "[explanation1]",
            "item2": "[explanation2]",
            ...
        }}
        ```
        """
        try:
            result = self.llm.invoke(prompt)
            item_explanation_dict = extract_and_convert_dict(result.content)
            item_explanation_dict = {
                exact_match_entity_type(item_in_result, item_list): explanation for item_in_result, explanation in item_explanation_dict.items()
            }
            return item_explanation_dict
        except:
            return {item: item for item in item_list}

    def retrieve_tools_from_candidates(self, query: str, candidate_tools: List[str], top_k: int = 5, explain_item: bool = False):
        if explain_item:
            query = self.explain_item(query)
        
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        tool_embeddings = []
        available_tools = []
        for tool_name in candidate_tools:
            if tool_name in self.tool_embedding_cache.keys():
                tool_embeddings.append(self.tool_embedding_cache[tool_name])
                available_tools.append(tool_name)
        
        if len(tool_embeddings) == 0:
            return [], []
        
        tool_embeddings = np.array(tool_embeddings)
        
        if np.isnan(query_embedding).any() or np.isnan(tool_embeddings).any():
            print("Warning: Embedding contains NaN.")
            return [], []
            
        similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        _tools = list(zip(available_tools, similarities))
        
        sorted_tools = sorted(_tools, key=lambda x: x[1], reverse=True)
        
        k = min(top_k, len(sorted_tools))
        if k == 0:
            return [], []

        top_k_tools, top_k_scores = zip(*sorted_tools[:k])
        
        return list(top_k_tools), list(top_k_scores)
