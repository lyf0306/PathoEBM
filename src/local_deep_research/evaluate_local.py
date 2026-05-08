import asyncio
import json
import logging
import os
import pickle
import time
from multiprocessing import Lock, Pool
from typing import Optional

import pandas as pd

from .config import settings
from .search_system import AdvancedSearchSystem


file_lock = Lock()


async def agent_infer(query: str, timeout_seconds: int = 1800) -> Optional[str]:
    """"""
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        
        system = AdvancedSearchSystem(
            max_iterations=2, questions_per_iteration=4, is_report=False
        )

        # initialize system
        await system.initialize()

        print("Welcome to the Advanced Research System")
        print("Type 'quit' to exit")

        start_time = time.time()
        results = await system.analyze_topic(query)
        elapsed_time = time.time() - start_time
        # print("agent_infer results:", results)
        print("agent_infer results current_knowledge:", results["current_knowledge"])
        if elapsed_time > timeout_seconds:
            logger.warning(f"Query took longer than {timeout_seconds} seconds")
            return "Sorry, the query took too long to process. Please try again with a more specific question."

        answer = results["current_knowledge"]
        return answer
    except Exception as e:
        logger.error(f"Error in agent_infer: {str(e)}")
        return f"Error processing query: {str(e)}"
    finally:
        # ensure system resources are cleaned up correctly
        if system and hasattr(system, "cleanup"):
            try:
                await system.cleanup()
            except Exception:
                pass
        # give async tasks a little time to complete cleanup
        await asyncio.sleep(0.00001)


def process_query(args):
    """process single query and save result immediately"""
    i, query, save_path = args
    try:
        print(f"Processing query {i}: {query}")
        start_time = time.time()

        
        answer = asyncio.run(agent_infer(query))

        elapsed_time = time.time() - start_time
        data = f"question id: {i} \nquestion: {query} \nanswer: {answer} \nprocessing time: {elapsed_time:.2f}s\n\n"

       
        with file_lock:
            with open(save_path, "a") as f:
                f.write(data)
                f.flush()  

        print(f"Completed query {i} in {elapsed_time:.2f}s")
        return i, query, answer
    except Exception as e:
        print(f"Error processing query {i}: {e}")
        with file_lock:
            with open(save_path, "a") as f:
                data = f"question id: {i} \nquestion: {query} \nERROR: {str(e)}\n\n"
                f.write(data)
                f.flush()
        return i, query, (f"ERROR: {str(e)}", "")


def run_evaluation(
    dataset_name="litqa",
    save_name="agent_answers_local.txt",
    num_processes=5,
    use_indices=False,
    indices_path=None,
):
    """run evaluation main function""" 

    # use relative path, for others to use
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    query_list = []

    
    indices = None
    if use_indices and indices_path:
        try:
            with open(indices_path, "rb") as f:
                indices = pickle.load(f)
        except Exception as e:
            print(f"Error loading indices: {e}")
            return None

   
    if dataset_name == "litqa":
        xlsx_path = os.path.join(
            project_root, "benchmark", "LitQA", "LitQA2_250424.xlsx"
        )
        save_path = os.path.join(project_root, "benchmark", "LitQA", save_name)

        
        if not os.path.exists(xlsx_path):
            print(f"dataset file not found: {xlsx_path}")
            return None

        # clear or create result file
        with open(save_path, "w") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_excel(xlsx_path)

        for i, row in df.iterrows():
            question = row["question"]
            choices = ""
            for choice in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                if not pd.isnull(row[f"choice_{choice}"]):
                    choice_text = row[f"choice_{choice}"]
                    choices += f"\n{choice}. {choice_text}"
            query = f"[single choice] {question} {choices}"
            query_list.append((i, query, save_path))

    

    elif dataset_name == "trqa_db_short":
        csv_path = os.path.join(
            project_root, "benchmark", "TRQA_db_short_ans", "TRQA-db-641.csv"
        )
        save_path = os.path.join(
            project_root, "benchmark", "TRQA_db_short_ans", save_name
        )

        if not os.path.exists(csv_path):
            print(f"dataset file not found: {csv_path}")
            return None

        with open(save_path, "w") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            question = row["Question"]
            query = f"[short answer] {question}"
            query_list.append((i, query, save_path))

    elif dataset_name == "trqa_lit_choice":
        csv_path = os.path.join(
            project_root,
            "benchmark",
            "TRQA_lit_choice",
            "TRQA-lit-choice-172-coreset.csv",
        )
        save_path = os.path.join(
            project_root, "benchmark", "TRQA_lit_choice", save_name
        )

        if not os.path.exists(csv_path):
            print(f"dataset file not found: {csv_path}")
            return None

        with open(save_path, "w") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            question = row["Question"]
            options_str = row["Options"]
            
            try:
                options = json.loads(options_str)
                choices = ""
                for key, value in options.items():
                    choices += f"\n{key}. {value}"
                # query = f"[multiple choice] (you must give at least one answer) {question} {choices}"
                query = f"[multiple choice] {question} {choices}"
                query_list.append((i, query, save_path))
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse options for question {i}: {options_str}"
                )
                continue

    elif dataset_name == "trqa_lit_short":
        csv_path = os.path.join(
            project_root,
            "benchmark",
            "TRQA_lit_short_ans",
            "TRQA-lit-short-answer-1108.csv",
        )
        save_path = os.path.join(
            project_root, "benchmark", "TRQA_lit_short_ans", save_name
        )

        if not os.path.exists(csv_path):
            print(f"dataset file not found: {csv_path}")
            return None

        with open(save_path, "w") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            question = row["Question"]
            query = f"[short answer] {question}"
            query_list.append((i, query, save_path))

    


    else:
        print(f"unsupported dataset: {dataset_name}")
        print(
            "supported datasets: litqa, trqa_db_short, trqa_lit_choice, trqa_lit_short"
        )
        return None

    
    if use_indices and indices:
        query_list = [data for data in query_list if data[0] in indices]

    print(f"Found {len(query_list)} questions to process")
    print(f"Results will be saved to: {save_path}")
    print(f"Starting parallel processing with {num_processes} processes")

   
    with Pool(processes=num_processes) as pool:
        results = pool.map_async(process_query, query_list)
        results.wait()

    print("All queries processed!")

    # 汇总所有结果
    with open(save_path, "a") as f:
        f.write(f"\nAll processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Results saved to: {save_path}")
    return save_path


def print_dataset_info():
    """print available dataset information"""
    print("\n=== available dataset information ===")
    print("1. litqa - literature question dataset (choice question)")
    print("2. trqa_db_short - TRQA database short answer dataset (short answer)")
    print("3. trqa_lit_choice - TRQA literature choice question dataset (multiple choice)")
    print("4. trqa_lit_short - TRQA literature short answer dataset (short answer)")
    print("\ninput format:")
    print("- choice question: [single choice] or [multiple choice] + question + options")
    print("- short answer: [short answer] + question")
    print("\noutput format:")
    print("- question id: [id]")
    print("- question: [question]")
    print("- answer: [answer]")
    print("- processing time: [processing time]")
    print("=" * 50)


if __name__ == "__main__":
    
    print_dataset_info()

    # you can modify these parameters to run different evaluations
    # dataset_name = "trqa_lit_choice"  # optional: "litqa",  "trqa_db_short", "trqa_lit_choice", "trqa_lit_short"
    # save_name = "agent_answers_test.txt"

    dataset_name = "trqa_lit_choice"  # optional: "litqa",  "trqa_db_short", "trqa_lit_choice", "trqa_lit_short"
    save_name = "agent_answers_test.txt"
    num_processes = 5 # adjust the number of processes according to your system
    use_indices = False  # whether to use index file
    indices_path = None  # index file path

    result_path = run_evaluation(
        dataset_name=dataset_name,
        save_name=save_name,
        num_processes=num_processes,
        use_indices=use_indices,
        indices_path=indices_path,
    )

    if result_path:
        print(f"Evaluation completed. Results in: {result_path}")

    print("\nhow to use other datasets:")
    print("modify the dataset_name variable to one of the following values:")
    print("- 'litqa' - literature question dataset (choice question)")
    print("- 'trqa_db_short' - database short answer dataset (short answer)")
    print("- 'trqa_lit_choice' - literature choice question dataset (multiple choice)")
    print("- 'trqa_lit_short' - literature short answer dataset (short answer)")
