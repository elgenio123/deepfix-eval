import os
import json
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
            timeout=15.0, # Add a timeout to prevent freezing indefinitely
        )
        # Using a free model on OpenRouter
        self.model = os.getenv("OPENROUTER_MODEL")

        self.valid_types = [it.value for it in IssueType]
        self.valid_severities = [s.value for s in Severity]

    def parse(self, report: str) -> List[Dict[str, str]]:
        """
        Extracts issue type and severity from a raw text report using an LLM.
        """
        if not self.api_key:
            print("Cannot parse using LLM without OPENROUTER_API_KEY. Returning empty list.")
            return []

        system_prompt = f"""You are an expert AI that parses natural language reports from a diagnostic tool into structured data.
Your task is to extract the identified issues and their severities.

The valid issue types are exactly the following:
{', '.join(self.valid_types)}

If the report mentions an issue that does not correspond to any of these known types, use the exact key "unknown_issue".

The valid severities are: {', '.join(self.valid_severities)}. If not specified in the text, default to MEDIUM.

Respond ONLY with a JSON array of objects. Each object must have exactly two keys: "type" (string) and "severity" (string).
Do not include any explanation or markdown formatting like ```json. Just output the raw JSON array.
"""

        try:
            print(f"      -> Sending request to OpenRouter ({self.model})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse the following report and extract the issues as a JSON array:\n\n{report}"}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting just in case the model ignores instructions
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            content = content.strip()
            
            parsed_data = json.loads(content)
            
            # Basic validation
            predictions = []
            for item in parsed_data:
                if "type" in item and "severity" in item:
                    pred_type = item["type"]
                    # Map to unknown_issue if not a valid type
                    if pred_type not in self.valid_types and pred_type != "unknown_issue":
                        pred_type = "unknown_issue"
                        
                    pred_severity = item["severity"].upper()
                    if pred_severity not in self.valid_severities:
                        pred_severity = "MEDIUM"
                        
                    # Avoid duplicates
                    if pred_type not in [p["type"] for p in predictions]:
                        predictions.append({
                            "type": pred_type,
                            "severity": pred_severity
                        })
            return predictions
            
        except Exception as e:
            print(f"Error parsing with LLM: {e}")
            return []

# if __name__ == "__main__":
#     parser = OutputParser()
#     sample_report = "1. [HIGH] Class Imbalance: The target distribution is skewed."
#     print("Testing parser with sample report...")
#     print(parser.parse(sample_report))
