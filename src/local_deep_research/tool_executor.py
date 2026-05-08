import asyncio
import json
import logging
import os
import urllib.parse
from datetime import datetime
from langchain_openai import ChatOpenAI
# from .connect_mcp import connect_mcp
from .utils import exact_match_entity_type, extract_and_convert_dict
from .utilties.search_utilities import (
    write_log_process_safe, write_json_log_process_safe
)

class ToolExecutor:
    """
    Execute the tools according to the tool calling input.
    """
    def __init__(self, mcp_client, error_log_path, llm_light):
        self.mcp_client = mcp_client
        self.tool_map = mcp_client.mcp_tool_map
        self.error_log_path = error_log_path
        self.execution_failed_log_path = os.path.join(os.path.dirname(error_log_path), "execution_failed_tools.json")
        self.llm_light = llm_light
        
    async def execute_tool_with_timeout(self, tool_invoke_info, timeout=180.0, max_retries=3):
        for attempt in range(max_retries):
            try:
                tool_calling_result = await asyncio.wait_for(
                    self.execute_tool(tool_invoke_info),
                    timeout=timeout
                )
                return tool_calling_result
            except asyncio.TimeoutError:
                print(f"Attempt {attempt + 1}/{max_retries} timed out when executing tool after {timeout}s")
                log_msg = f"[{datetime.now().isoformat()}] Attempt {attempt + 1}/{max_retries} timed out when executing tool after {timeout}s. Tool_invoke_info: {tool_invoke_info}"
                write_log_process_safe(self.error_log_path, log_msg)
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 1}/{max_retries}...")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed when executing tool with error: {e}")
                log_msg = f"[{datetime.now().isoformat()}] Attempt {attempt + 1}/{max_retries} failed when executing tool with error: {e}. Tool_invoke_info: {tool_invoke_info}"
                write_log_process_safe(self.error_log_path, log_msg)
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 1}/{max_retries}...")
                else:
                    logging.error(f"Max retries reached when executing tool. Giving up. Tool_invoke_info: {tool_invoke_info}")
                    log_msg = f"[{datetime.now().isoformat()}] Max retries reached when executing tool. Giving up. Tool_invoke_info: {tool_invoke_info}. Error: {e}\n"
                    write_log_process_safe(self.error_log_path, log_msg)
                    
                    # Safe get tool name
                    t_name = tool_invoke_info.get("tool") or tool_invoke_info.get("tool_name", "Unknown")
                    
                    tool_calling_result = {
                        "content": f"Error: Failed after {max_retries} attempts when executing tool: {e}. Tool_invoke_info: {tool_invoke_info}",
                        "tool_name": t_name,
                        "toolsuite": None,
                        "success": False,
                    }
                    return tool_calling_result
        log_msg = f"[{datetime.now().isoformat()}] Max retries reached when executing tool. Giving up. Tool_invoke_info: {tool_invoke_info}"
        write_log_process_safe(self.error_log_path, log_msg)
        
        t_name = tool_invoke_info.get("tool") or tool_invoke_info.get("tool_name", "Unknown")
        
        tool_calling_result = {
            "content": f"Error: Failed after {max_retries} attempts when executing tool. Tool_invoke_info: {tool_invoke_info}",
            "tool_name": t_name,
            "toolsuite": None,
            "success": False,
        }
        
        return tool_calling_result

    async def execute_tool(self, tool_invoke_info):
        """
        Execute single tool.
        """
        tool = None
        
        # 1. Get tool from tool_map
        try:
            tool_name_key = tool_invoke_info.get("tool") or tool_invoke_info.get("tool_name")
            
            if tool_name_key:
                tool = self.tool_map.get(tool_name_key)
            else:
                logging.error(f"Missing 'tool' or 'tool_name' key in invoke info: {tool_invoke_info}")
                
        except Exception as e:
            logging.error(f"Error accessing tool_map: {e}")
            
        # 2. If tool not found, log and set error result
        if tool is None:
            try:
                t_name = tool_invoke_info.get("tool") or tool_invoke_info.get("tool_name", "Unknown")
                logging.info(
                    f"Tool {t_name} not found in the tool pool(with {len(self.tool_map)} tools available)."
                )
                result = f"Error: Tool {t_name} not found in the tool pool(with {len(self.tool_map)} tools available)."
            except Exception as e:
                logging.error(f"Error logging tool not found: {e}")
        else:
            # 3. Prepare tool input and invoke tool
            try:
                tool_input = tool_invoke_info["tool_input"]
            except Exception as e:
                logging.error(f"Error extracting tool_input: {e}")
                raise e
            try:
                logging.info(
                    f"Executing tool: {getattr(tool, 'name', 'Unknown')} with input: {tool_input}"
                )
            except Exception as e:
                logging.error(f"Error logging tool execution: {e}")
                raise e
            # 4. Invoke tool with try/except
            try:
                result = await tool.ainvoke(tool_input)
            except Exception as e:
                logging.error(f"Error invoking tool {getattr(tool, 'name', 'Unknown')}, tool_input: {tool_input}, tool_invoke_info: {tool_invoke_info}, error: {e}")
                raise e
                
        # 5. Get tool_name with try/except
        try:
            tool_name = tool_invoke_info.get("tool") or tool_invoke_info.get("tool_name", "")
            if tool_name:
                try:
                    tool_name = exact_match_entity_type(tool_name, list(self.tool_map.keys()))
                except Exception as e:
                    logging.error(f"Error in exact_match_entity_type: {e}")
                    tool_name = tool_invoke_info.get("tool") or tool_invoke_info.get("tool_name", "")
            else:
                raise Exception("Tool name is empty")
        except Exception as e:
            logging.error(f"Error extracting tool_name: {e}")
            raise e
            
        # 6. Get tool_source with try/except
        try:
            tool_source = self.mcp_client.tool2source.get(tool_name, "Unknown")
        except Exception as e:
            logging.error(f"Error extracting tool_source: {e}")
            raise e
            
        # 7. Build tool_calling_result with try/except
        try:
            tool_calling_result = {
                "content": str(result),
                "tool_name": tool_name,
                "toolsuite": tool_source,
                "tool_input": tool_input,  # <--- 【修改点 1】: 将 tool_input 传递下去，用于生成 URL
                "success": self.judge_output_is_meaningful(result)
            }
        except Exception as e:
            logging.error(f"Error building tool_calling_result: {e}")
            raise e
            
        # 8. Extract additional info with try/except
        try:
            tool_calling_result = self.extract_additional_info(tool_calling_result)
        except Exception as e:
            logging.error(f"Error extracting additional info: {e}")
            # 不抛出异常，而是记录错误并继续，避免影响主流程
            pass
            
        return tool_calling_result

    def extract_additional_info(self, tool_calling_result):
        """
        Extract the additional information from the tool calling result.
        【修改点 2】: 全新的链接生成逻辑，支持 FDA, ClinicalTrials, NCBI 等
        """
        try:
            result = tool_calling_result.get("content", {})
            tool_name = tool_calling_result.get("tool_name", "").lower()
            tool_input = tool_calling_result.get("tool_input", {}) or {}

            # 确保 result 是字典 (有些工具返回的是 JSON 字符串)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass

            info_type = "Others"
            info_content = ""
            info_urls = []

            # 1. 保持原有的化学/Tavily 逻辑 (PubChem / PDB / Tavily)
            if isinstance(result, dict) and "SMILES" in result and result.get("SMILES"):
                smiles = result["SMILES"]
                info_type = "Compound"
                info_content = smiles
                info_urls = [f"https://pubchem.ncbi.nlm.nih.gov/#query={urllib.parse.quote(smiles)}"]
            elif isinstance(result, dict) and "pdb_id" in result and result.get("pdb_id"):
                pdb_id = result["pdb_id"]
                info_type = "Protein"
                info_content = pdb_id
                info_urls = [f"https://www.rcsb.org/structure/{urllib.parse.quote(pdb_id)}"]
            elif "tavily_search" in tool_name and isinstance(result, dict):
                info_urls = [each["url"] for each in result.get("results", [])]
                info_type = "URL"

            # 2. 【新增】FDA 工具链接生成
            elif "fda" in tool_name or "drug" in tool_name:
                info_type = "OpenFDA"
                # 尝试获取药名
                query_term = tool_input.get("drug_name") or tool_input.get("query") or "drug"
                info_urls = [f"https://open.fda.gov/apis/drug/label/?q={urllib.parse.quote(str(query_term))}"]
                info_content = f"FDA Label for {query_term}"

            # 3. 【新增】ClinicalTrials 链接生成
            elif "clinical" in tool_name or "studies" in tool_name:
                info_type = "ClinicalTrials"
                query_term = tool_input.get("query") or tool_input.get("condition") or "trials"
                info_urls = [f"https://clinicaltrials.gov/search?term={urllib.parse.quote(str(query_term))}"]
                info_content = "ClinicalTrials.gov Search"

            # 4. 【新增】NCBI/Gene 链接生成
            elif "ncbi" in tool_name or "gene" in tool_name:
                info_type = "NCBI"
                gene = tool_input.get("gene_name") or tool_input.get("name") or "gene"
                info_urls = [f"https://www.ncbi.nlm.nih.gov/gene/?term={urllib.parse.quote(str(gene))}"]
                info_content = f"NCBI Gene: {gene}"

            # 5. 【兜底】防止引用池为空
            elif tool_name: 
                # 生成一个伪协议链接，确保 SearchSystem 能记录这个来源
                safe_input = urllib.parse.quote(str(tool_input)[:50])
                info_urls = [f"tool://{tool_name}?input={safe_input}"]

            # 赋值回对象
            tool_calling_result["additional_info_type"] = info_type
            tool_calling_result["additional_info"] = info_content
            tool_calling_result["additional_urls"] = info_urls

        except Exception as e:
            logging.error(f"Error extracting additional info: {e}")
            tool_calling_result["additional_info_type"] = "Error"
            tool_calling_result["additional_urls"] = []

        return tool_calling_result
    
    def judge_output_is_meaningful(self, tool_response):
        if not isinstance(tool_response, str):
            try:
                tool_response = str(tool_response)
            except Exception as e:
                log_msg = f"[{datetime.now().isoformat()}] Tool response is not a string: {tool_response}"
                write_log_process_safe(self.error_log_path, log_msg)
                return False
        
        prompt = f"""
# Your Task
You are a senior data scientist. I currently have an MCP service with many tools. Please help me determine whether the tool execution was successful based on the tool's output.
# Tool calling result
Top 500 tokens of the full content: {tool_response[:500]}
# Instructions
1. The returned result may be a JSON-formatted string (part of the full tool calling result), null, or an empty string.
2. If the returned result contains words like "error", "404", "400", or "No data found", it indicates that the request failed.
# Output Format
Return data in python dict format as follows:
```python
{{
    "success": [success status] # True if the tool ran successfully with substantial information returned, False if it failed
}}
""" 
        try:
            response = self.llm_light.invoke(prompt) 
            result = extract_and_convert_dict(response.content)
            
            if isinstance(result, dict) and "success" in result:
                return result["success"]
            else:
                return False
        except Exception as e:
            logging.error(f"Error in judge_output_is_meaningful: {e}")
            return False

    def log_execution_failed_tool(self, tool_invoke_list, tool_calling_results):
        for tool_invoke_info, tool_calling_result in zip(tool_invoke_list, tool_calling_results):
            if tool_calling_result.get("success", False):
                continue
            
            error_info = tool_invoke_info.copy()
            error_info['tool_calling_result'] = tool_calling_result["content"]
            error_info['log_time'] = datetime.now().isoformat()
            write_json_log_process_safe(self.execution_failed_log_path, error_info)

    async def run(self, tool_invoke_list):
        """
        Execute the tools.
        """
        execute_tasks = [
            self.execute_tool_with_timeout(tool_invoke_info) for tool_invoke_info in tool_invoke_list
        ]
        
        tool_calling_results = await asyncio.gather(*execute_tasks)
        
        # Log failed tools
        self.log_execution_failed_tool(tool_invoke_list, tool_calling_results)
        
        return tool_calling_results