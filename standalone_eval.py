import os
import json
import argparse
import pandas as pd
import config
from config import RAW_RESULTS_DIR, TaskType, DATA_DIR, OUTPUT_DIR
from output_parser import OutputParser
from evaluator import Evaluator
from reporter import Reporter
from visualizer import Visualizer

def reconstruct_metadata(pipeline_ids):
    """Reconstruct ground truths and task types from the outputs/data directory."""
    print(f"Reconstructing metadata for {len(pipeline_ids)} pipelines from {DATA_DIR}...")
    ground_truths = {}
    task_types = {}
    
    for pid in pipeline_ids:
        path = os.path.join(DATA_DIR, pid)
        if not os.path.exists(path):
            print(f"   -> Warning: Data directory not found for {pid}")
            continue
        
        # Load Ground Truth
        gt_path = os.path.join(path, "ground_truth.json")
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                ground_truths[pid] = json.load(f)
        
        # Infer Task Type from train.csv
        train_path = os.path.join(path, "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            # In our dataset generator: 
            # BAF (Classification) has int64 target.
            # Lending Club (Regression) has float64 target.
            if pd.api.types.is_float_dtype(df['target']) and df['target'].nunique() > 20:
                task_types[pid] = TaskType.REGRESSION
            else:
                task_types[pid] = TaskType.CLASSIFICATION
                
    return ground_truths, task_types

def evaluate_version(raw_reports, ground_truths, task_types, version_name, output_dir):
    """Run parsing and evaluation for a specific version."""
    print(f"\n>>> Evaluating Version: {version_name}")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Output Parsing
    print(f"1. Parsing unstructured reports for {version_name}...")
    parser = OutputParser()
    predictions = {}
    
    pipeline_ids = list(raw_reports.keys())
    if not pipeline_ids:
        print(f"No reports found for {version_name}.")
        return None

    # Use 20 batches for 100 pipelines to stay within API limits and GEMINI.md guidelines
    num_requests = 20
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
    print(f"2. Computing metrics for {version_name}...")
    evaluator = Evaluator()
    results_all = evaluator.evaluate(ground_truths, predictions)
    
    clf_pids = [pid for pid, tt in task_types.items() if tt == TaskType.CLASSIFICATION and pid in predictions]
    results_clf = evaluator.evaluate({pid: ground_truths[pid] for pid in clf_pids}, {pid: predictions.get(pid, []) for pid in clf_pids})
    
    reg_pids = [pid for pid, tt in task_types.items() if tt == TaskType.REGRESSION and pid in predictions]
    results_reg = evaluator.evaluate({pid: ground_truths[pid] for pid in reg_pids}, {pid: predictions.get(pid, []) for pid in reg_pids})
    
    results = {
        "overall": results_all,
        "classification": results_clf,
        "regression": results_reg
    }

    # 3. Reporting & Visualization
    print(f"3. Generating outputs for {version_name}...")
    Reporter(output_dir=reports_dir).report(results, prefix=f"eval_{version_name}")
    Visualizer(output_dir=reports_dir).visualize([], results=results)
    
    return results

def run_standalone_eval(local_file=None, online_file=None, metadata_file=None):
    """
    Run standalone evaluation, supporting comparative analysis if both files are provided.
    """
    # Load Reports
    local_reports = {}
    online_reports = {}
    
    if local_file and os.path.exists(local_file):
        with open(local_file, 'r') as f:
            local_reports = json.load(f)
    
    if online_file and os.path.exists(online_file):
        with open(online_file, 'r') as f:
            online_reports = json.load(f)
            
    if not local_reports and not online_reports:
        print("Error: No valid report files provided or files do not exist.")
        return

    # Load/Reconstruct Metadata
    all_pids = list(set(list(local_reports.keys()) + list(online_reports.keys())))
    ground_truths = {}
    task_types = {}
    
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        ground_truths = metadata.get("ground_truths", {})
        task_types_raw = metadata.get("task_types", {})
        task_types = {pid: TaskType(tt) for pid, tt in task_types_raw.items()}
    else:
        # Reconstruct from outputs/data
        ground_truths, task_types = reconstruct_metadata(all_pids)

    # Filter to only include pids we actually have metadata for
    valid_pids = set(ground_truths.keys())
    local_reports = {pid: r for pid, r in local_reports.items() if pid in valid_pids}
    online_reports = {pid: r for pid, r in online_reports.items() if pid in valid_pids}

    # Evaluate Versions
    local_results = None
    online_results = None
    
    if local_reports:
        local_results = evaluate_version(local_reports, ground_truths, task_types, "local", "outputs_comparative/local")
        
    if online_reports:
        online_results = evaluate_version(online_reports, ground_truths, task_types, "online", "outputs_comparative/online")

    # Comparative Analysis
    if local_results and online_results:
        print("\n>>> Generating Comparative Analysis...")
        comp_dir = "outputs_comparative/reports"
        os.makedirs(comp_dir, exist_ok=True)
        Reporter(output_dir=comp_dir).report_comparative(online_results, local_results, "Online", "Local")
        
        print("\nCOMPARATIVE SUMMARY")
        for metric in ["f1_macro", "f1_micro"]:
            val_online = online_results['overall']['detection'][metric]
            val_local = local_results['overall']['detection'][metric]
            print(f"{metric}: Online={val_online:.4f}, Local={val_local:.4f} (Delta: {val_local-val_online:+.4f})")
    
    print("\nStandalone evaluation complete. Check outputs_comparative/ for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFix Standalone Evaluation")
    parser.add_argument("--local", type=str, help="Local raw reports JSON file")
    parser.add_argument("--online", type=str, help="Online raw reports JSON file")
    parser.add_argument("--metadata", type=str, help="Metadata JSON file (optional, will reconstruct if missing)")
    args = parser.parse_args()
    
    # Default paths for comparative if not provided
    l_file = args.local or "outputs_comparative/local/raw/deepfix_raw_outputs.json"
    o_file = args.online or "outputs_comparative/online/raw/deepfix_raw_outputs.json"
    
    run_standalone_eval(local_file=l_file, online_file=o_file, metadata_file=args.metadata)
