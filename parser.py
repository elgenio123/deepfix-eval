import re
from typing import List, Dict, Any
from config import ISSUE_REGISTRY, IssueType, Severity

class OutputParser:
    """
    Parses DeepFix's natural language output into structured predictions.
    Supports a rule-based approach using synonym matching.
    """
    
    def __init__(self):
        # Build an inverted index of synonyms to issue types
        self.synonym_map = {}
        for issue_type, meta in ISSUE_REGISTRY.items():
            self.synonym_map[issue_type.value] = issue_type.value
            # add normalized issue type name
            normalized_name = issue_type.value.replace("_", " ").lower()
            self.synonym_map[normalized_name] = issue_type.value
            
            for syn in meta.synonyms:
                self.synonym_map[syn.lower()] = issue_type.value

    def parse(self, report: str) -> List[Dict[str, str]]:
        """
        Extracts issue type and severity from a raw text report.
        """
        predictions = []
        report_lower = report.lower()
        
        # Rule-based parsing
        # Find lines that seem to list an issue
        lines = report.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Detect severity
            severity_val = None
            if "high" in line_lower:
                severity_val = Severity.HIGH.value
            elif "medium" in line_lower:
                severity_val = Severity.MEDIUM.value
            elif "low" in line_lower:
                severity_val = Severity.LOW.value
                
            # Try to match an issue based on synonyms
            detected_type = None
            for phrase, issue_val in self.synonym_map.items():
                # Word boundary match to avoid partial matches
                if re.search(r'\b' + re.escape(phrase) + r'\b', line_lower):
                    detected_type = issue_val
                    break
                    
            if detected_type:
                # If we couldn't find severity, fallback to default
                if not severity_val:
                    severity_val = ISSUE_REGISTRY[IssueType(detected_type)].default_severity.value
                    
                # Avoid duplicates per pipeline
                if detected_type not in [p["type"] for p in predictions]:
                    predictions.append({
                        "type": detected_type,
                        "severity": severity_val
                    })
                    
        return predictions

if __name__ == "__main__":
    parser = OutputParser()
    sample_report = "1. [HIGH] Class Imbalance: The target distribution is skewed."
    print(parser.parse(sample_report))
