import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ================= 配置区域 =================
# 这里必须与你生成 pkl 时使用的模型路径一致
EMBEDDING_MODEL_PATH = "/root/Qwen3-Embedding-4B"

# 你的新模板文件路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLINICAL_PKL_PATH = os.path.join(CURRENT_DIR, "clinical_templates.pkl")

# ===========================================

print(f"🔄 [TemplateAgent] Initializing...")
print(f"   - Loading Local Embedding Model: {EMBEDDING_MODEL_PATH}")

# 1. 加载本地 Embedding 模型 (全局加载一次，避免重复开销)
try:
    _embed_model = SentenceTransformer(
        EMBEDDING_MODEL_PATH, 
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("   - Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    _embed_model = None

# 2. 加载临床模板数据
_clinical_templates = None

def load_templates():
    global _clinical_templates
    if _clinical_templates is not None:
        return

    if not os.path.exists(CLINICAL_PKL_PATH):
        print(f"❌ Error: Clinical templates file not found at {CLINICAL_PKL_PATH}")
        return
    
    try:
        with open(CLINICAL_PKL_PATH, "rb") as f:
            _clinical_templates = pickle.load(f)
        print(f"   - Loaded {len(_clinical_templates['large']['value_list'])} clinical templates.")
    except Exception as e:
        print(f"❌ Error loading pkl file: {e}")

# 初始化加载
load_templates()

def get_embedding(text):
    """使用本地模型生成向量"""
    if _embed_model is None:
        return np.zeros(2560) # 返回空向量防止崩溃
    
    try:
        # normalize_embeddings=True 确保可以直接计算余弦相似度
        embedding = _embed_model.encode(text, normalize_embeddings=True)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(2560)

def retrieve_large_template(query, research_mode="audit"):
    """
    检索最匹配的临床思维模板
    参数 research_mode 在这里保留是为了兼容接口，
    但无论传什么，我们现在只返回临床模板。
    """
    # 确保数据已加载
    if _clinical_templates is None:
        return "[(1, 'Search', 'Search for NCCN guidelines regarding the treatment plan.')]"

    try:
        # 1. 计算 Query 的向量
        query_embedding = get_embedding(query)
        
        # 2. 计算与所有模板的相似度 (点积)
        # clinical_templates['large']['embeddings'] 是 (N, 2560)
        # query_embedding 是 (2560,)
        scores = np.dot(_clinical_templates["large"]["embeddings"], query_embedding)
        
        # 3. 找到分数最高的索引
        best_idx = np.argmax(scores)
        
        # 4. 返回对应的模板字符串
        return _clinical_templates["large"]["value_list"][best_idx]
        
    except Exception as e:
        print(f"Error retrieving template: {e}")
        # 发生错误时的保底模板
        return _clinical_templates["large"]["value_list"][0]

def retrieve_small_template(query):
    """保留此函数以防其他模块调用，逻辑同上"""
    return retrieve_large_template(query)