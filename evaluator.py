from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from config import IssueType

class Evaluator:
    """
    Computes rigorous evaluation metrics based on Raha, MLDebugger, REIN, and Data Validation for ML.
    """
    
    def __init__(self):
        self.issue_types = [it.value for it in IssueType]
        
    def _binarize(self, issues: List[Dict[str, str]]) -> np.ndarray:
        """Convert a list of issues into a binary vector over all issue types."""
        vec = np.zeros(len(self.issue_types), dtype=int)
        types_present = {issue["type"] for issue in issues}
        for i, it in enumerate(self.issue_types):
            if it in types_present:
                vec[i] = 1
        return vec

    def evaluate(self, ground_truths: Dict[str, List[Dict[str, Any]]], predictions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth across all pipelines.
        """
        pipeline_ids = list(ground_truths.keys())
        
        y_true_all = []
        y_pred_all = []
        
        # Pipeline-level metrics
        find_one_count = 0
        find_all_count = 0
        detection_rate_count = 0
        
        # Alert Quality Metrics
        total_fps = 0
        total_tns = 0
        extra_predictions = []
        
        for pid in pipeline_ids:
            gt = ground_truths[pid]
            pred = predictions.get(pid, [])
            
            y_t = self._binarize(gt)
            y_p = self._binarize(pred)
            
            y_true_all.append(y_t)
            y_pred_all.append(y_p)
            
            # MLDebugger Pipeline-Level Metrics
            # Detection Rate: At least one correct prediction
            correct_preds = np.logical_and(y_t, y_p)
            if np.any(y_p): # Model fired an alert
                detection_rate_count += 1
                
            if np.any(correct_preds):
                find_one_count += 1
                
            if np.array_equal(y_t, np.logical_or(y_t, correct_preds)) and np.sum(y_t) == np.sum(correct_preds) and np.sum(y_t) > 0:
                # all true positives found and no false negatives
                find_all_count += 1
                
            # Data Validation for ML Metrics
            fp_vec = np.logical_and(np.logical_not(y_t), y_p)
            tn_vec = np.logical_and(np.logical_not(y_t), np.logical_not(y_p))
            
            total_fps += np.sum(fp_vec)
            total_tns += np.sum(tn_vec)
            
            extra_preds = max(0, len(pred) - len(gt))
            extra_predictions.append(extra_preds)
            
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        n_pipelines = len(pipeline_ids)
        
        # Raha Detection Metrics
        # Macro averaging treats all classes equally
        precision_macro = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        
        # Micro averaging treats all instances equally
        precision_micro = precision_score(y_true_all, y_pred_all, average='micro', zero_division=0)
        recall_micro = recall_score(y_true_all, y_pred_all, average='micro', zero_division=0)
        f1_micro = f1_score(y_true_all, y_pred_all, average='micro', zero_division=0)
        
        # Per-Issue Metrics
        per_issue_metrics = {}
        for i, it in enumerate(self.issue_types):
            p = precision_score(y_true_all[:, i], y_pred_all[:, i], zero_division=0)
            r = recall_score(y_true_all[:, i], y_pred_all[:, i], zero_division=0)
            f = f1_score(y_true_all[:, i], y_pred_all[:, i], zero_division=0)
            per_issue_metrics[it] = {"precision": p, "recall": r, "f1": f}
            
        # Alert Quality Metrics
        fpr = total_fps / (total_fps + total_tns) if (total_fps + total_tns) > 0 else 0.0
        over_detection_penalty = float(np.mean(extra_predictions))
        
        results = {
            "detection": {
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
                "f1_micro": float(f1_micro)
            },
            "pipeline_level": {
                "detection_rate": float(detection_rate_count / n_pipelines),
                "find_one_rate": float(find_one_count / n_pipelines),
                "find_all_rate": float(find_all_count / n_pipelines)
            },
            "alert_quality": {
                "false_positive_rate": float(fpr),
                "over_detection_penalty": float(over_detection_penalty)
            },
            "per_issue": per_issue_metrics
        }
        
        return results
