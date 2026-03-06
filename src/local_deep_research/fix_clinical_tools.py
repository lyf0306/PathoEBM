"""
临床证据合成框架 - 工具配置修复脚本

使用方法:
1. 将此脚本放在 src/local_deep_research/ 目录
2. 运行: python fix_clinical_tools.py
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# ================== 配置区 ==================
CLINICAL_TOOLS = [
    "tavily_search",           # 通用搜索
    "search_papers",           # 文献检索
    "get_general_info_by_disease_name",  # 疾病信息
]

# 临床专用实体类型（简化版）
CLINICAL_ENTITIES = [
    "Disease",           # 疾病/癌症类型
    "Drug/Drug class",   # 药物
    "Clinical",          # 临床决策点
    "Biomarker",         # 分子标志物
    "Therapeutic target" # 治疗靶点
]

# ================== 1. 生成新的 tool_info.xlsx ==================
def create_clinical_tool_info():
    """创建临床工具配置表"""
    
    data = {
        'tool_name': [
            'tavily_search',
            'tavily_search', 
            'search_papers',
            'search_papers',
            'get_general_info_by_disease_name',
        ],
        'input_entity': [
            'Clinical',      # 审计临床决策
            'Drug/Drug class',  # 查询药物信息
            'Disease',       # 文献检索疾病
            'Biomarker',     # 检索分子标志物相关文献
            'Disease',       # 获取疾病基本信息
        ],
        'output_entity': [
            'Clinical',
            'Drug/Drug class',
            'Clinical',
            'Clinical',
            'Disease',
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 保存到 cache_data 目录
    cache_dir = Path(__file__).parent / "cache_data"
    cache_dir.mkdir(exist_ok=True)
    
    output_path = cache_dir / "tool_info.xlsx"
    
    # 使用 ExcelWriter 创建单 sheet
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='clinical_tools', index=False)
    
    print(f"✅ 已生成临床工具配置: {output_path}")
    return output_path

# ================== 2. 清理 Embedding 缓存 ==================
def clean_embedding_cache(mcp_tool_map):
    """移除不在 CLINICAL_TOOLS 中的工具 embedding"""
    
    cache_path = Path(__file__).parent / "cache_data" / "tool_desc_embedding.pkl"
    
    if not cache_path.exists():
        print("⚠️ 未找到 embedding 缓存，将在运行时自动生成")
        return
    
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        # 检测格式并转换
        if isinstance(cache, dict):
            if "tool_embeddings" in cache:
                # 新格式 -> 旧格式
                old_cache = {}
                for name, emb in zip(cache["tool_names"], cache["tool_embeddings"]):
                    if name in CLINICAL_TOOLS:
                        old_cache[name] = emb
                cache = old_cache
            else:
                # 旧格式直接过滤
                cache = {k: v for k, v in cache.items() if k in CLINICAL_TOOLS}
        
        # 保存清理后的缓存
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"✅ 已清理 embedding 缓存，保留 {len(cache)} 个工具")
        
    except Exception as e:
        print(f"❌ 清理缓存失败: {e}")
        print("   建议删除缓存文件，让系统重新生成")

# ================== 3. 修复实体分类 ==================
def update_utils_file():
    """提示修改 utils.py 中的生物实体列表"""
    
    utils_path = Path(__file__).parent / "utils.py"
    
    print("\n" + "="*60)
    print("⚠️ 需要手动修改 utils.py:")
    print("="*60)
    print(f"文件路径: {utils_path}")
    print("\n将 biological_entities 列表替换为:\n")
    print("biological_entities = [")
    for entity in CLINICAL_ENTITIES:
        print(f'    "{entity}",')
    print("]\n")
    print("="*60)

# ================== 4. 检查模板系统 ==================
def check_template_system():
    """检查临床模板是否正确配置"""
    
    template_path = Path(__file__).parent / "tools" / "template" / "clinical_templates.pkl"
    
    if not template_path.exists():
        print("\n⚠️ 未找到 clinical_templates.pkl")
        print("   请确保已运行模板生成脚本")
        return
    
    try:
        with open(template_path, 'rb') as f:
            templates = pickle.load(f)
        
        if "large" in templates and "embeddings" in templates["large"]:
            num_templates = len(templates["large"]["value_list"])
            print(f"✅ 临床模板系统正常 ({num_templates} 个模板)")
        else:
            print("❌ 模板格式错误")
            
    except Exception as e:
        print(f"❌ 模板加载失败: {e}")

# ================== 主函数 ==================
def main():
    print("\n" + "🏥"*20)
    print("  子宫内膜癌临床证据合成框架 - 配置修复工具")
    print("🏥"*20 + "\n")
    
    # 步骤1: 生成工具配置
    print("\n[1/4] 生成临床工具配置表...")
    create_clinical_tool_info()
    
    # 步骤2: 清理embedding (需要 mcp 连接才能获取 tool_map)
    print("\n[2/4] 检查 Embedding 缓存...")
    print("⚠️ 建议删除旧缓存，让系统自动重建:")
    cache_path = Path(__file__).parent / "cache_data" / "tool_desc_embedding.pkl"
    print(f"   rm {cache_path}")
    
    # 步骤3: 提示修改实体分类
    print("\n[3/4] 实体分类系统...")
    update_utils_file()
    
    # 步骤4: 检查模板
    print("\n[4/4] 检查临床模板...")
    check_template_system()
    
    print("\n" + "="*60)
    print("✅ 配置检查完成!")
    print("="*60)
    print("\n后续步骤:")
    print("1. 按照提示修改 utils.py")
    print("2. 删除旧的 embedding 缓存")
    print("3. 运行系统，让它自动生成新缓存")
    print("4. 测试审计功能: python main.py --mode audit --context_file test.txt")

if __name__ == "__main__":
    main()