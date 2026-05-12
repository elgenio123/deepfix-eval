import numpy as np
from visualizer import Visualizer
from config import IssueType
from dataset_generator import PipelineInstance, TaskType

def test_per_class_bar():
    visualizer = Visualizer()
    
    # Mock results
    per_issue = {}
    for it in IssueType:
        per_issue[it.value] = {
            "precision": np.random.random(),
            "recall": np.random.random(),
            "f1": np.random.random()
        }
        
    results = {
        "overall": {
            "per_issue": per_issue
        }
    }
    
    # Mock pipelines (empty list for simplicity as we are only testing plot_per_class_metrics)
    pipelines = []
    
    print("Testing plot_per_class_metrics...")
    visualizer.plot_per_class_metrics(results)
    print("Check outputs/reports/f1_per_issue_type.png")

if __name__ == "__main__":
    test_per_class_bar()
