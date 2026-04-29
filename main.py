import argparse
import random
import numpy as np

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
    predictions = {}
    ground_truths = {}
    task_types = {}
    
    for pipeline in pipelines:
        pid = pipeline.pipeline_id
        ground_truths[pid] = pipeline.ground_truth_issues
        task_types[pid] = pipeline.task_type
        
        raw_report = raw_outputs[pid]
        predictions[pid] = parser.parse(raw_report)
        # print(predictions)
        
    # 4. Evaluation
    print("4. Computing metrics...")
    evaluator = Evaluator()
    
    results_all = evaluator.evaluate(ground_truths, predictions)
    
    clf_pids = [pid for pid, tt in task_types.items() if tt == TaskType.CLASSIFICATION]
    gt_clf = {pid: ground_truths[pid] for pid in clf_pids}
    pred_clf = {pid: predictions[pid] for pid in clf_pids}
    results_clf = evaluator.evaluate(gt_clf, pred_clf)
    
    reg_pids = [pid for pid, tt in task_types.items() if tt == TaskType.REGRESSION]
    gt_reg = {pid: ground_truths[pid] for pid in reg_pids}
    pred_reg = {pid: predictions[pid] for pid in reg_pids}
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
    visualizer.visualize(pipelines)
    
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
