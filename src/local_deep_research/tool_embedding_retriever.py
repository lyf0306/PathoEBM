import json
from typing import List
import os
import pickle as pkl
import asyncio
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .utils import extract_and_convert_dict, exact_match_entity_type

# === 新增：引入本地模型库 ===
import torch
from sentence_transformers import SentenceTransformer

class LocalQwenEmbedding:
    # 你的本地模型路径
    MODEL_PATH = "/root/Qwen3-Embedding-4B"
    
    def __init__(self):
        print(f"🔄 [Embedding] Loading Local Model: {self.MODEL_PATH}")
        try:
            self.model = SentenceTransformer(
                self.MODEL_PATH, 
                trust_remote_code=True, 
                device="cuda:1" if torch.cuda.is_available() else "cpu"
            )
            print("✅ [Embedding] Model loaded successfully.")
        except Exception as e:
            print(f"❌ [Embedding] Error loading model: {e}")
            self.model = None
    
    def embed_query(self, query: str):
        if self.model is None:
            return np.zeros(2560)
        # normalize_embeddings=True 对余弦相似度检索至关重要
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding # SentenceTransformer 返回的是 numpy array
    
    def embed_documents(self, query_list: List[str]):
        if self.model is None:
            return [np.zeros(2560) for _ in query_list]
        embedding_list = self.model.encode(query_list, normalize_embeddings=True)
        return embedding_list

class ToolEmbeddingRetriever:
    
    def __init__(self, llm, mcp_tool_client, embedding_api_key: str, embedding_cache: str, available_tools: list = None):
        self.llm = llm
        self.mcp_tool_client = mcp_tool_client
        self.mcp_tool_map = mcp_tool_client.mcp_tool_map
        
        # === 修改：使用本地模型替代 API ===
        # 原来的 embedding_api_key 在这里会被忽略
        self.embedding_model = LocalQwenEmbedding()
        
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
                    # 这是我们用脚本生成的 tool_desc_embedding.pkl 新格式
                    # 需要将其转换回 {tool_name: embedding} 的字典格式供 Retriever 使用
                    self.tool_embedding_cache = {}
                    names = data["tool_names"]
                    embeddings = data["tool_embeddings"]
                    for name, emb in zip(names, embeddings):
                        self.tool_embedding_cache[name] = emb
                    print(f"✅ Loaded {len(self.tool_embedding_cache)} tools from new PKL format.")
                else:
                    # 旧格式
                    self.tool_embedding_cache = data
                    print(f"✅ Loaded {len(self.tool_embedding_cache)} tools from legacy PKL format.")
                    
            except Exception as e:
                print(f"❌ Error loading PKL: {e}")
                self.tool_embedding_cache = {}
        else:
            self.tool_embedding_cache = {}
            
        cached_embedding_num = len(self.tool_embedding_cache)
        
        # Check if mcp_tool_map is initialized
        if not hasattr(self, 'mcp_tool_map'):
            print("Warning: mcp_tool_map not initialized yet")
            return
            
        # 如果缓存缺失，尝试补充 (但通常我们已经手动重构了 pkl)
        for tool_name, tool in tqdm(self.mcp_tool_map.items()):
            if tool_name in self.tool_embedding_cache:
                continue
                
            # Check if tool has description
            if not hasattr(tool, 'description'):
                print(f"Warning: tool {tool_name} has no description")
                continue
                
            tool_func_desc = tool.description.split('Args')[0]
            embedding = self.embedding_model.embed_query(tool_func_desc)
            
            # Only cache if embedding was successful
            if embedding is not None:
                self.tool_embedding_cache[tool_name] = embedding
            else:
                print(f"Warning: Failed to generate embedding for {tool_name}")
        
        # Save cache if new embeddings were added
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
        """
        Simple RAG. Retrieve the most relevant tools for a given query using embedding similarity.
        """
        # Get query embedding
        if explain_item:
            query = self.explain_item(query)
            
        query_embedding = self.embedding_model.embed_query(query)
        # Ensure 2D array
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Check for NaN values
        if np.isnan(query_embedding).any() or (len(self.all_tool_embedding) > 0 and np.isnan(self.all_tool_embedding).any()):
            print("Warning: Embedding contains NaN.")
            return [], []
            
        if len(self.all_tool_embedding) == 0:
            return [], []

        # Calculate cosine similarity scores
        similarities = cosine_similarity(query_embedding, self.all_tool_embedding)[0]
        _tools = list(zip(self.tool_name_list, similarities))
            
        # Sort tools by similarity score and get top k
        sorted_tools = sorted(_tools, key=lambda x: x[1], reverse=True)
        
        # Safe slicing
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
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarity scores
        tool_embeddings = []
        available_tools = []
        for tool_name in candidate_tools:
            if tool_name in self.tool_embedding_cache.keys():
                tool_embeddings.append(self.tool_embedding_cache[tool_name])
                available_tools.append(tool_name)
        
        if len(tool_embeddings) == 0:
            return [], []
        
        tool_embeddings = np.array(tool_embeddings)
        
        # Check for NaN values
        if np.isnan(query_embedding).any() or np.isnan(tool_embeddings).any():
            print("Warning: Embedding contains NaN.")
            return [], []
            
        similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        _tools = list(zip(available_tools, similarities))
        
        # Sort tools by similarity score and get top k
        sorted_tools = sorted(_tools, key=lambda x: x[1], reverse=True)
        
        k = min(top_k, len(sorted_tools))
        if k == 0:
            return [], []

        top_k_tools, top_k_scores = zip(*sorted_tools[:k])
        
        return list(top_k_tools), list(top_k_scores)