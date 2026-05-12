import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from config import IssueType, Severity

# Load environment variables
load_dotenv()

class OutputParser:
    """
    Parses DeepFix's natural language output into structured predictions using an LLM via OpenRouter.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not found in .env")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key or "dummy_key", # Provide a dummy key to avoid initialization errors if not set yet
            timeout=60.0, # Increased timeout for larger batches
        )
        # Using a light model on OpenRouter by default
        self.model = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5-8b")

        self.valid_types = [it.value for it in IssueType]
        self.valid_severities = [s.value for s in Severity]
        
        # Regex to strip ANSI escape codes
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def _strip_ansi(self, text: str) -> str:
        """Removes ANSI escape sequences (colors, formatting) from text."""
        return self.ansi_escape.sub('', text)

    def _extract_json(self, content: str) -> str:
        """Robustly extracts JSON content from a potentially messy LLM response."""
        content = content.strip()
        # Find the first '{' or '[' and the last '}' or ']'
        start_idx = content.find('{')
        if start_idx == -1:
            start_idx = content.find('[')
            
        end_idx = content.rfind('}')
        if end_idx == -1:
            end_idx = content.rfind(']')
            
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return content[start_idx:end_idx+1]
        return content

    def parse(self, report: str) -> List[Dict[str, str]]:
        """
        Extracts issue type and severity from a raw text report using an LLM.
        """
        if not self.api_key:
            print("Cannot parse using LLM without OPENROUTER_API_KEY. Returning empty list.")
            return []

        clean_report = self._strip_ansi(report)

        system_prompt = f"""You are an expert AI that parses natural language reports from a diagnostic tool into structured data.
Your task is to extract the identified issues and their severities.

The valid issue types are exactly the following:
{', '.join(self.valid_types)}

If the report mentions an issue that does not correspond to any of these known types, use the exact key "unknown_issue".

The valid severities are: {', '.join(self.valid_severities)}. If not specified in the text, default to MEDIUM.

Respond ONLY with a JSON array of objects. Each object must have exactly two keys: "type" (string) and "severity" (string).
"""

        try:
            print(f"      -> Sending request to OpenRouter ({self.model})...")
            # Some models on OpenRouter support 'json_object' response format
            extra_kwargs = {}
            if any(m in self.model.lower() for m in ["gemini", "gpt-4", "gpt-3.5", "claude-3"]):
                extra_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse the following report and extract the issues as a JSON array:\n\n{clean_report}"}
                ],
                temperature=0.0,
                **extra_kwargs
            )
            
            raw_content = response.choices[0].message.content
            content = self._extract_json(raw_content)
            parsed_data = json.loads(content)
            
            # If the model wrapped the array in an object
            if isinstance(parsed_data, dict):
                for key in ["issues", "predictions", "data", "array"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        parsed_data = parsed_data[key]
                        break
                if isinstance(parsed_data, dict):
                    parsed_data = list(parsed_data.values()) if all(isinstance(v, dict) for v in parsed_data.values()) else []

            if not isinstance(parsed_data, list):
                return []

            # Basic validation
            predictions = []
            for item in parsed_data:
                if isinstance(item, dict) and "type" in item and "severity" in item:
                    pred_type = item["type"]
                    if pred_type not in self.valid_types and pred_type != "unknown_issue":
                        pred_type = "unknown_issue"
                        
                    pred_severity = str(item["severity"]).upper()
                    if pred_severity not in self.valid_severities:
                        pred_severity = "MEDIUM"
                        
                    if pred_type not in [p["type"] for p in predictions]:
                        predictions.append({
                            "type": pred_type,
                            "severity": pred_severity
                        })
            return predictions
            
        except Exception as e:
            print(f"Error parsing with LLM: {e}")
            return []

    def parse_batch(self, reports: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Extracts issue types and severities from multiple reports in a single LLM call.
        """
        if not self.api_key:
            print("Cannot parse using LLM without OPENROUTER_API_KEY. Returning empty dict.")
            return {pid: [] for pid in reports.keys()}

        # Clean all reports
        clean_reports = {pid: self._strip_ansi(report) for pid, report in reports.items()}

        system_prompt = f"""You are an expert AI that parses natural language reports from a diagnostic tool into structured data.
Your task is to extract the identified issues and their severities for MULTIPLE reports.

The valid issue types are exactly the following:
{', '.join(self.valid_types)}

If a report mentions an issue that does not correspond to any of these known types, use the exact key "unknown_issue".

The valid severities are: {', '.join(self.valid_severities)}. If not specified in the text, default to MEDIUM.

The input will be a JSON object where keys are pipeline IDs and values are the text reports.
Respond ONLY with a single JSON object where keys are the same pipeline IDs and values are JSON arrays of objects. 
Each object in the array must have exactly two keys: "type" (string) and "severity" (string).
"""

        try:
            print(f"      -> Sending batch request to OpenRouter ({self.model})...")
            
            extra_kwargs = {}
            if any(m in self.model.lower() for m in ["gemini", "gpt-4", "gpt-3.5", "claude-3"]):
                extra_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse the following reports and extract the issues as a JSON object:\n\n{json.dumps(clean_reports, indent=2)}"}
                ],
                temperature=0.0,
                max_tokens=4096,
                **extra_kwargs
            )
            
            raw_content = response.choices[0].message.content
            content = self._extract_json(raw_content)
            parsed_batch_data = json.loads(content)
            
            # If the model wrapped the result in a top-level key
            if "results" in parsed_batch_data and isinstance(parsed_batch_data["results"], dict):
                parsed_batch_data = parsed_batch_data["results"]
            
            all_predictions = {}
            for pid in reports.keys():
                predictions = []
                reports_data = parsed_batch_data.get(pid)
                
                if not isinstance(reports_data, list):
                    all_predictions[pid] = []
                    continue

                for item in reports_data:
                    if isinstance(item, dict) and "type" in item and "severity" in item:
                        pred_type = item["type"]
                        if pred_type not in self.valid_types and pred_type != "unknown_issue":
                            pred_type = "unknown_issue"
                            
                        pred_severity = str(item["severity"]).upper()
                        if pred_severity not in self.valid_severities:
                            pred_severity = "MEDIUM"
                            
                        if pred_type not in [p["type"] for p in predictions]:
                            predictions.append({
                                "type": pred_type,
                                "severity": pred_severity
                            })
                all_predictions[pid] = predictions
                
            return all_predictions
            
        except Exception as e:
            print(f"Error parsing batch with LLM: {e}")
            # If the error is likely truncation, print a hint
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                print("      Hint: The LLM response was likely truncated. Try a smaller batch size or a model with a larger context window.")
            return {pid: [] for pid in reports.keys()}

# if __name__ == "__main__":
#     parser = OutputParser()
#     sample_report = "1. [HIGH] Class Imbalance: The target distribution is skewed."
#     print("Testing parser with sample report...")
#     print(parser.parse(sample_report))
