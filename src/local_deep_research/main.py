import asyncio
import logging
import sys
import os

from .config import settings, get_local_model
from .search_system import AdvancedSearchSystem

# 配置基础日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_local_model_health(timeout: float = 2.0) -> bool:
    """
    检测本地模型(vLLM)是否可用。
    
    Args:
        timeout: 连接超时时间(秒)，设短一点以免卡顿
    """
    print("🔍 Checking local model connectivity...", end=" ", flush=True)
    try:
        # 获取模型实例，强制设置短超时
        # 注意：这里需要 config.py 中的 get_local_model 支持传参或我们在下面临时设置
        llm = get_local_model(temperature=0.1)
        
        # 覆盖 request_timeout (如果 langchain 版本支持)
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = timeout
            
        # 发送一个极简的测试请求
        llm.invoke("Hi") 
        print("✅ ONLINE")
        return True
    except Exception as e:
        # 捕获所有连接错误（ConnectionRefused, Timeout 等）
        print(f"❌ OFFLINE (Error: {str(e)[:50]}...)")
        return False

def read_context_from_file(file_path: str) -> str:
    """Helper: 从文件读取治疗方案内容"""
    try:
        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading context file: {e}")
        return ""

async def run_evidence_update(treatment_context: str):
    """
    执行单一任务：临床证据更新 (自动切换模式)
    """
    # === 核心逻辑：自动降级 ===
    if check_local_model_health():
        print("🚀 Using Local vLLM Model (Free & Private).")
        current_mode = "local"
    else:
        print("⚠️ Local model unavailable. Switching to Cloud API (DeepSeek).")
        current_mode = "deepseek"
    
    print(f"\n🔄 Clinical Evidence Update System Activated.")
    print(f"   Context Length: {len(treatment_context)} characters")
    print("   Targeting Sources: PubMed (2024+), ClinicalTrials.gov, FDA\n")

    # 初始化系统，传入自动选择的 mode
    system = AdvancedSearchSystem(
        max_iterations=settings.detailed.iteration, 
        questions_per_iteration=settings.detailed.questions_per_iteration,
        is_report=True,
        treatment_context=treatment_context,
        using_model=current_mode  # <--- 传入自动判断的结果
    )

    try:
        await system.initialize()
        
        query = "Verify this treatment plan against 2024-2025 guidelines and latest clinical trials."
        
        results = await system.analyze_topic(query)
        
        print(f"\n✅ Evidence verification task completed.")
        
        if results.get("final_report"):
             print("\n" + "="*60)
             print("   FINAL EVIDENCE UPDATE REPORT   ")
             print("="*60 + "\n")
             print(results["final_report"])
             
             report_path = "evidence_update_report.md"
             with open(report_path, "w", encoding="utf-8") as f:
                 f.write(results["final_report"])
             print(f"\n📄 Report saved to: {os.path.abspath(report_path)}")

    except Exception as e:
        logger.error(f"Run failed: {e}")
        print(f"\n❌ Error during execution: {e}")

async def main():
    """
    主程序入口
    """
    print("==================================================")
    print("   OriGene Clinical Evidence Validator (Auto-Hybrid)")
    print("==================================================")
    print("Strategy: Try Local Model first -> Fallback to Cloud API")
    print("Type 'quit' to exit at any time.")

    while True:
        print("\n--------------------------------------------------")
        print("Select Input Method:")
        print("1) Paste Treatment Plan Text")
        print("2) Load Plan from File (.txt)")
        
        choice = input("\nEnter number (1 or 2): ").strip()
        
        if choice.lower() == 'quit':
            break

        treatment_context = ""

        if choice == "2":
            path = input("Enter file path: ").strip()
            if path.lower() == 'quit': break
            treatment_context = read_context_from_file(path)
        elif choice == "1":
            print("\n👇 Please paste the Clinical Treatment Plan below.")
            print("Type 'END' on a new line when finished:\n")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            treatment_context = "\n".join(lines)
        else:
            print("Invalid selection. Please enter 1 or 2.")
            continue
        
        if not treatment_context.strip():
            print("❌ Empty context provided. Please try again.")
            continue

        # 执行核心任务
        await run_evidence_update(treatment_context)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")