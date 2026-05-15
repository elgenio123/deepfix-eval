import os
import random
import numpy as np
import json
import config

from dataset_generator import DatasetGenerator
from deepfix_runner import DeepFixRunner
from output_parser import OutputParser
from evaluator import Evaluator
from reporter import Reporter
from visualizer import Visualizer

def run_eval_for_version(pipelines, api_url: str, version_name: str, seed: int, output_dir: str):
    print(f"\nRunning Evaluation for Version: {version_name}")
    print(f"Results will be saved to: {output_dir}")
    
    # Update config for the runner (it still uses config.RAW_RESULTS_DIR)
    config.RAW_RESULTS_DIR = os.path.join(output_dir, "raw")
    os.makedirs(config.RAW_RESULTS_DIR, exist_ok=True)
    
    runner = DeepFixRunner(api_url=api_url)
    raw_outputs = runner.run_all(pipelines)
    
    parser = OutputParser()
    ground_truths = {p.pipeline_id: p.ground_truth_issues for p in pipelines}
    task_types = {p.pipeline_id: p.task_type for p in pipelines}
    
    num_requests = 5
    pipeline_ids = [p.pipeline_id for p in pipelines]
    k, m = divmod(len(pipeline_ids), num_requests)
    batches_ids = [pipeline_ids[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_requests)]
    
    predictions = {}
    for i, batch_id_list in enumerate(batches_ids):
        if not batch_id_list: continue
        batch_reports = {pid: raw_outputs[pid] for pid in batch_id_list}
        batch_predictions = parser.parse_batch(batch_reports)
        predictions.update(batch_predictions)
        
    evaluator = Evaluator()
    results_all = evaluator.evaluate(ground_truths, predictions)
    
    clf_pids = [pid for pid, tt in task_types.items() if tt == config.TaskType.CLASSIFICATION]
    results_clf = evaluator.evaluate({pid: ground_truths[pid] for pid in clf_pids}, {pid: predictions.get(pid, []) for pid in clf_pids})
    
    reg_pids = [pid for pid, tt in task_types.items() if tt == config.TaskType.REGRESSION]
    results_reg = evaluator.evaluate({pid: ground_truths[pid] for pid in reg_pids}, {pid: predictions.get(pid, []) for pid in reg_pids})
    
    results = {"overall": results_all, "classification": results_clf, "regression": results_reg}
    
    # Use explicit output_dir for Reporter and Visualizer
    reports_dir = os.path.join(output_dir, "reports")
    Reporter(output_dir=reports_dir).report(results, prefix=f"eval_{version_name}_seed_{seed}")
    Visualizer(output_dir=reports_dir).visualize(pipelines, results=results)
    
    return results

def main():
    seed = config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    
    base_output_dir = "outputs_comparative"
    online_dir = os.path.join(base_output_dir, "online")
    local_dir = os.path.join(base_output_dir, "local")
    
    pipelines = DatasetGenerator(seed=seed).generate_dataset(config.PIPELINE_COUNTS_PER_TASK)
    
    online_results = run_eval_for_version(pipelines, None, "online", seed, online_dir)
    local_results = run_eval_for_version(pipelines, "http://localhost:8844/v2/analyse", "local", seed, local_dir)
    
    # Generate comparative report in the base_output_dir
    comp_reports_dir = os.path.join(base_output_dir, "reports")
    Reporter(output_dir=comp_reports_dir).report_comparative(online_results, local_results, "Online", "Local")
    
    print("\nCOMPARATIVE SUMMARY")
    for metric in ["f1_macro", "f1_micro"]:
        val_online = online_results['overall']['detection'][metric]
        val_local = local_results['overall']['detection'][metric]
        print(f"{metric}: Online={val_online:.4f}, Local={val_local:.4f} (Delta: {val_local-val_online:+.4f})")
        
    for metric in ["detection_rate", "find_all_rate"]:
        val_online = online_results['overall']['pipeline_level'][metric]
        val_local = local_results['overall']['pipeline_level'][metric]
        print(f"{metric}: Online={val_online:.4f}, Local={val_local:.4f} (Delta: {val_local-val_online:+.4f})")

if __name__ == "__main__":
    main()
