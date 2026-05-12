import argparse
import random
import numpy as np
import json

from config import PIPELINE_COUNTS_PER_TASK, RANDOM_SEED, TaskType
from dataset_generator import DatasetGenerator
from deepfix_runner import DeepFixRunner
from output_parser import OutputParser
from evaluator import Evaluator
from reporter import Reporter
from visualizer import Visualizer

def run_experiment(seed: int = RANDOM_SEED):
    """Run a single end-to-end experiment pipeline."""
    print(f"Starting experiment with seed={seed}...")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Dataset Generation
    print("1. Sampling dataset from seaborn")
    generator = DatasetGenerator(seed=seed)
    pipelines = generator.generate_dataset(PIPELINE_COUNTS_PER_TASK)
    print(f"   Generated {len(pipelines)} pipelines.")
    
    # 2. DeepFix Execution
    print("2. Running DeepFix SDK...")
    runner = DeepFixRunner()
    raw_outputs = runner.run_all(pipelines)
    
    # 3. Output Parsing
    print("3. Parsing unstructured reports...")
    parser = OutputParser()
    ground_truths = {p.pipeline_id: p.ground_truth_issues for p in pipelines}
    task_types = {p.pipeline_id: p.task_type for p in pipelines}
    
    # Batch processing to optimize API usage (exactly 5 requests as per GEMINI.md)
    num_requests = 5
    pipeline_ids = [p.pipeline_id for p in pipelines]
    if not pipeline_ids:
        print("No pipelines generated.")
        return None
        
    k, m = divmod(len(pipeline_ids), num_requests)
    batches_ids = [pipeline_ids[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_requests)]
    
    predictions = {}
    for i, batch_id_list in enumerate(batches_ids):
        if not batch_id_list:
            continue
        print(f"   -> Batch {i+1}/{num_requests} ({len(batch_id_list)} pipelines)")
        batch_reports = {pid: raw_outputs[pid] for pid in batch_id_list}
        batch_predictions = parser.parse_batch(batch_reports)
        predictions.update(batch_predictions)
        
    # 4. Evaluation
    print("4. Computing metrics...")
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
    
    # 5. Reporting
    print("5. Generating reports...")
    reporter = Reporter()
    reporter.report(results, prefix=f"experiment_seed_{seed}")
    
    # 6. Visualization
    print("6. Generating visualizations...")
    visualizer = Visualizer()
    visualizer.visualize(pipelines, results=results)
    
    return results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="DeepFix Evaluation Framework")
    argparser.add_argument("--runs", type=int, default=1, help="Number of repetitions for stability analysis")
    args = argparser.parse_args()

    if args.runs == 1:
        run_experiment()
    else:
        print(f"Running {args.runs} repetitions...")
        all_results = []
        for i in range(args.runs):
            res = run_experiment(seed=RANDOM_SEED + i)
            all_results.append(res)
            
        print("Completed multiple runs. (Stability aggregation is left for future extensions)")
