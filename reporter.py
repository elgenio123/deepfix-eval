import os
import json
from typing import Dict, Any

from config import REPORTS_DIR

class Reporter:
    """
    Generates structured markdown reports and saves JSON evaluation results.
    """
    
    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
    def generate_markdown_report(self, results: Dict[str, Any], filepath: str):
        """
        Generates a Markdown file with tables for the evaluation metrics.
        """
        lines = []
        lines.append("# DeepFix Evaluation Report\n")
        
        for task_subset, subset_results in results.items():
            lines.append(f"## {task_subset.title()} Metrics")
            lines.append("### Global Metrics Table")
            lines.append("| Metric Category | Metric | Value |")
            lines.append("|---|---|---|")
            
            for category in ["detection", "pipeline_level", "alert_quality"]:
                for metric, value in subset_results[category].items():
                    formatted_metric = metric.replace("_", " ").title()
                    lines.append(f"| {category.title()} | {formatted_metric} | {value:.4f} |")
                    
            lines.append("\n### Per-Issue Metrics")
            lines.append("| Issue Type | Precision | Recall | F1 Score |")
            lines.append("|---|---|---|---|")
            
            for issue_type, metrics in subset_results["per_issue"].items():
                lines.append(f"| {issue_type} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
                
            lines.append("\n")
            
        lines.append("## Visualizations")
        lines.append("### Residual Scatterplot")
        lines.append("![Residual Scatterplot](residual_scatterplot.png)\n")
        lines.append("### Error by Group")
        lines.append("![Error by Group](error_by_group.png)\n")
        lines.append("### Group-wise Confusion Matrices")
        lines.append("![Group-wise Confusion Matrices](groupwise_confusion_matrices.png)\n")
            
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
            
    def report(self, results: Dict[str, Any], prefix: str = "evaluation"):
        """
        Saves results to both JSON and Markdown formats.
        """
        json_path = os.path.join(REPORTS_DIR, f"{prefix}_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
            
        md_path = os.path.join(REPORTS_DIR, f"{prefix}_report.md")
        self.generate_markdown_report(results, md_path)
        
        print(f"Reports generated successfully at {REPORTS_DIR}")
        print(f"- JSON: {json_path}")
        print(f"- Markdown: {md_path}")
