"""
OriClinical 纯权威数据源初始化脚本
功能: 
1. 仅配置 NIH(NCBI/ClinicalTrials)、FDA、OpenTargets、ClinVar 等权威数据库工具。
2. 剔除所有通用搜索引擎 (Tavily/Jina)，确保无来源不明信息。
3. 自动更新 utils.py 和 tool_info.xlsx。
"""

import pandas as pd
import re
import time
from pathlib import Path

# ================== 1. 临床实体定义 ==================
# 系统的“词汇表”，LLM 会将用户问题中的关键词映射为这些实体
CLINICAL_ENTITIES = [
    "Clinical",           # 临床试验、综合查询
    "Drug/Drug class",    # 药物信息、FDA标签
    "Disease",            # 疾病定义、相关文献
    "Biomarker",          # 基因、突变、分子标志物
]

# ================== 2. 权威工具白名单 (Authoritative Only) ==================
# 以下工具名均已基于你提供的 OrigeneMCP 源码进行核实
TOOL_DATA = {
    'tool_name': [
        # === A. 临床试验 (NIH ClinicalTrials.gov) ===
        # 来源: tools/clinicaltrials/server.py
        'get_studies',                                # 搜索临床试验 (按条件)
        'get_study',                                  # 获取特定试验详情 (按NCT ID)
        
        # === B. FDA 药物监管信息 (OpenFDA) ===
        # 来源: tools/tooluniverse/data/fda_drug_labeling_tools.json
        'get_indications_by_drug_name',               # 官方适应症
        'get_warnings_by_drug_name',                  # 黑框警告/严重警告
        'get_contraindications_by_drug_name',         # 禁忌症
        'get_clinical_pharmacology_by_drug_name',     # 临床药理学数据
        'get_dosage_and_storage_information_by_drug_name', # 剂量与存储
        
        # === C. 权威文献证据 (OpenTargets -> PubMed/PMC) ===
        # 来源: tools/tooluniverse/data/opentarget_tools.json
        # 替代 Tavily 搜索，仅检索与疾病或药物强关联的学术文献
        'get_publications_by_disease_efoId',          # 疾病相关权威文献
        'get_publications_by_drug_chemblId',          # 药物相关权威文献
        
        # === D. 疾病与基因基础 (NCBI & OpenTargets & ClinVar) ===
        # 来源: tools/tooluniverse/data/opentarget_tools.json
        'get_disease_id_description_by_name',         # 疾病官方定义 (EFO Ontology)
        
        # 来源: tools/ncbi/server.py
        'get_gene_metadata_by_gene_name',             # 基因官方元数据 (RefSeq)
        
        # 来源: tools/dbsearch/server.py
        'clinvar_query_variant_significance',         # 突变致病性评级 (ClinVar)
    ],
    
    'input_entity': [
        # --- Clinical Trials ---
        'Disease',            # 查该疾病有哪些试验 (LLM需构造 query={"cond": "disease"})
        'Clinical',           # 查特定试验详情 (通常输入 NCT ID)
        
        # --- FDA ---
        'Drug/Drug class',    # 查适应症
        'Drug/Drug class',    # 查警告
        'Drug/Drug class',    # 查禁忌
        'Drug/Drug class',    # 查药理
        'Drug/Drug class',    # 查剂量
        
        # --- Literature (OpenTargets) ---
        'Disease',            # 查疾病文献
        'Drug/Drug class',    # 查药物文献
        
        # --- Basic Info ---
        'Disease',            # 查疾病定义
        'Biomarker',          # 查基因信息
        'Biomarker',          # 查突变意义
    ],
    
    'output_entity': [
        'Clinical',           # 试验列表
        'Clinical',           # 试验详情
        
        'Drug/Drug class',    # 适应症文本
        'Drug/Drug class',    # 警告文本
        'Drug/Drug class',    # 禁忌文本
        'Drug/Drug class',    # 药理文本
        'Drug/Drug class',    # 剂量文本
        
        'Clinical',           # 文献列表 (视为临床证据)
        'Clinical',           # 文献列表
        
        'Disease',            # 疾病描述
        'Biomarker',          # 基因数据
        'Clinical',           # 致病性评级
    ],
    
    'tool_description': [
        'Search for recruiting/completed clinical trials (NIH).',
        'Get full details of a clinical trial by NCT ID.',
        'Get FDA approved indications and usage.',
        'Get FDA boxed warnings and precautions.',
        'Get FDA contraindications.',
        'Get FDA clinical pharmacology data.',
        'Get FDA dosage and administration info.',
        'Get authoritative publications linking disease to targets.',
        'Get authoritative publications linking drug to targets.',
        'Get standard disease definition from OpenTargets.',
        'Get official gene metadata from NCBI RefSeq.',
        'Get clinical significance of a variant from ClinVar.',
    ]
}

# ================== 3. 执行逻辑 ==================

def create_tool_excel():
    """生成 Excel 配置文件"""
    print("📋 [1/3] 正在生成权威工具配置表...")
    
    # 自动查找路径
    base_dirs = [Path("src/local_deep_research"), Path("local_deep_research"), Path(".")]
    target_dir = None
    for d in base_dirs:
        if (d / "cache_data").exists() or d.name == "local_deep_research":
            target_dir = d / "cache_data"
            break
            
    if not target_dir:
        # 如果没找到，默认创建在 src 下
        target_dir = Path("src/local_deep_research/cache_data")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "tool_info.xlsx"
    
    df = pd.DataFrame(TOOL_DATA)
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet1 是代码默认读取的
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            # 备份一个 sheet
            df.to_excel(writer, sheet_name='authoritative_tools', index=False)
        print(f"✅ 成功写入: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 写入 Excel 失败: {e}")
        return False

def update_utils():
    """更新 utils.py 中的实体列表"""
    print("📝 [2/3] 正在同步实体定义到 utils.py...")
    
    # 查找 utils.py
    search_paths = [
        Path("src/local_deep_research/utils.py"),
        Path("local_deep_research/utils.py"),
        Path("utils.py")
    ]
    
    target_file = None
    for p in search_paths:
        if p.exists():
            target_file = p
            break
            
    if not target_file:
        print("❌ 未找到 utils.py，请手动更新 biological_entities 列表。")
        return

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 构造新的列表字符串
        new_list_str = "biological_entities = [\n"
        for ent in CLINICAL_ENTITIES:
            new_list_str += f'    "{ent}",\n'
        new_list_str += "]"
        
        # 正则替换
        pattern = r"biological_entities\s*=\s*\[[^\]]*\]"
        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, new_list_str, content, flags=re.DOTALL)
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ 已更新文件: {target_file}")
        else:
            print("⚠️ 无法在 utils.py 中定位 biological_entities，请检查代码格式。")
            
    except Exception as e:
        print(f"❌ 更新 utils.py 失败: {e}")

def clean_cache():
    """清理 Embedding 缓存"""
    print("🧹 [3/3] 正在清理旧的 Embedding 缓存...")
    
    paths_to_clean = [
        Path("src/local_deep_research/cache_data/tool_desc_embedding.pkl"),
        Path("local_deep_research/cache_data/tool_desc_embedding.pkl"),
        Path("cache_data/tool_desc_embedding.pkl")
    ]
    
    cleaned = False
    for p in paths_to_clean:
        if p.exists():
            try:
                p.unlink()
                print(f"✅ 已删除: {p}")
                cleaned = True
            except Exception as e:
                print(f"❌ 删除失败 {p}: {e}")
    
    if not cleaned:
        print("ℹ️ 未发现旧缓存，无需清理。")

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 🏥 OriClinical 权威数据源模式初始化")
    print("="*50 + "\n")
    
    if create_tool_excel():
        update_utils()
        clean_cache()
        
        print("\n" + "="*50)
        print("🎉 初始化完成！")
        print("="*50)
        print("系统现在仅使用以下权威数据源:")
        print("  1. NIH ClinicalTrials.gov (临床试验)")
        print("  2. FDA OpenFDA (药物标签与监管)")
        print("  3. OpenTargets Platform (疾病与文献)")
        print("  4. NCBI RefSeq (基因信息)")
        print("  5. ClinVar (突变致病性)")
        print("\n请运行以下命令启动审计:")
        print("  python src/local_deep_research/main.py --mode audit --context_file your_plan.txt")