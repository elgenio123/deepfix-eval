import os
import json
import argparse
from config import RAW_RESULTS_DIR, TaskType
from output_parser import OutputParser
from evaluator import Evaluator
from reporter import Reporter
from visualizer import Visualizer

def run_standalone_eval(metadata_file="experiment_metadata.json", reports_file="deepfix_raw_outputs.json"):
    """
    Run parsing and evaluation using saved raw outputs and metadata.
    """
    print(f"Loading metadata from {metadata_file} and reports from {reports_file}...")
    
    metadata_path = os.path.join(RAW_RESULTS_DIR, metadata_file)
    reports_path = os.path.join(RAW_RESULTS_DIR, reports_file)
    
    if not os.path.exists(metadata_path) or not os.path.exists(reports_path):
        print(f"Error: Required files not found in {RAW_RESULTS_DIR}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(reports_path, 'r') as f:
        raw_reports = json.load(f)

    ground_truths = metadata["ground_truths"]
    task_types_raw = metadata["task_types"]
    task_types = {pid: TaskType(tt) for pid, tt in task_types_raw.items()}

    # 1. Output Parsing
    print("1. Parsing unstructured reports...")
    parser = OutputParser()
    predictions = {}
    
    # Split reports into 5 batches to optimize API requests (following GEMINI.md guidelines)
    num_requests = 25
    pipeline_ids = list(raw_reports.keys())
    if not pipeline_ids:
        print("No reports found to parse.")
        return
        
    k, m = divmod(len(pipeline_ids), num_requests)
    batches_ids = [pipeline_ids[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_requests)]
    
    for i, batch_id_list in enumerate(batches_ids):
        if not batch_id_list:
            continue
        print(f"   -> Batch {i+1}/{num_requests} ({len(batch_id_list)} pipelines)")
        batch_reports = {pid: raw_reports[pid] for pid in batch_id_list}
        batch_predictions = parser.parse_batch(batch_reports)
        predictions.update(batch_predictions)

    # 2. Evaluation
    print("2. Computing metrics...")
    evaluator = Evaluator()
    
    results_all = evaluator.evaluate(ground_truths, predictions)
    
    clf_pids = [pid for pid, tt in task_types.items() if tt == TaskType.CLASSIFICATION]
    gt_clf = {pid: ground_truths[pid] for pid in clf_pids}
    pred_clf = {pid: predictions.get(pid, []) for pid in clf_pids}
    results_clf = evaluator.evaluate(gt_clf, pred_clf)
    
    reg_pids = [pid for pid, tt in task_types.items() if tt == TaskType.REGRESSION]
    gt_reg = {pid: ground_truths[pid] for pid in reg_pids}
    pred_reg = {pid: predictions.get(pid, []) for pid in reg_pids}
    results_reg = evaluator.evaluate(gt_reg, pred_reg)
    
    results = {
        "overall": results_all,
        "classification": results_clf,
        "regression": results_reg
    }

    # 3. Reporting
    print("3. Generating reports...")
    reporter = Reporter()
    reporter.report(results, prefix="standalone_eval")
    
    # 4. Visualization
    # Note: Visualizer.visualize takes pipelines list for some plots.
    # We might need to mock or load PipelineInstance if we want all plots.
    # For now, we provide an empty list but pass results for detection plots.
    print("4. Generating visualizations...")
    visualizer = Visualizer()
    visualizer.visualize([], results=results)
    
    print("Standalone evaluation complete. Check outputs/ for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFix Standalone Evaluation")
    parser.add_argument("--metadata", type=str, default="experiment_metadata.json", help="Metadata JSON file")
    parser.add_argument("--reports", type=str, default="deepfix_raw_outputs.json", help="Raw reports JSON file")
    args = parser.parse_args()
    
    run_standalone_eval(metadata_file=args.metadata, reports_file=args.reports)
