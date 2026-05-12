from ast import Dict
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score
from typing import List

from config import REPORTS_DIR, TaskType
from dataset_generator import PipelineInstance

class Visualizer:
    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        # Set styling
        sns.set_theme(style="whitegrid")

    def visualize(self, pipelines: List[PipelineInstance], results = None):
        self.plot_residual_scatterplot(pipelines)
        self.plot_error_by_group(pipelines)
        self.plot_groupwise_confusion_matrices(pipelines)
        if results:
            self.plot_per_issue_metrics(results)

    def plot_per_issue_metrics(self, results):
        """Plot precision, recall, and f1 for each issue type using evaluation results."""
        # Use overall results for per-issue metrics
        data = results.get("overall", results).get("per_issue", {})
        if not data:
            return

        issue_types = list(data.keys())
        precisions = [data[it]["precision"] for it in issue_types]
        recalls = [data[it]["recall"] for it in issue_types]
        f1s = [data[it]["f1"] for it in issue_types]

        x = np.arange(len(issue_types))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - width, precisions, width, label='Precision', color='#1f77b4')
        ax.bar(x, recalls, width, label='Recall', color='#ff7f0e')
        ax.bar(x + width, f1s, width, label='F1 Score', color='#2ca02c')

        ax.set_ylabel('Score')
        ax.set_title('Detection Metrics per Issue Type')
        ax.set_xticks(x)
        ax.set_xticklabels(issue_types, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(REPORTS_DIR, "per_issue_metrics.png")
        plt.savefig(filepath)
        plt.close()

    def plot_residual_scatterplot(self, pipelines: List[PipelineInstance]):
        """Plot true vs. predicted (or error vs. predicted) for regression."""
        # Find regression pipelines that have a fitted model
        reg_pipelines = [p for p in pipelines if p.task_type == TaskType.REGRESSION and p.model is not None]
        if not reg_pipelines:
            return

        plt.figure(figsize=(10, 6))
        for pipeline in reg_pipelines:
            try:
                X_test = pipeline.test_data["X"]
                y_test = pipeline.test_data["y"]
                y_pred = pipeline.model.predict(X_test)
                residuals = y_test - y_pred
                
                # Plot residuals vs predicted
                plt.scatter(y_pred, residuals, alpha=0.5, label=f"Pipeline {pipeline.pipeline_id[:6]}")
            except Exception:
                continue

        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (True - Predicted)")
        plt.title("Residual Scatterplot for Regression Pipelines")
        # Do not show legend if too many pipelines
        if len(reg_pipelines) <= 5:
            plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(REPORTS_DIR, "residual_scatterplot.png")
        plt.savefig(filepath)
        plt.close()

    def plot_error_by_group(self, pipelines: List[PipelineInstance]):
        """Bar chart of mean absolute error or error rate per sensitive group."""
        group_errors = {"Classification": {0: [], 1: []}, "Regression": {0: [], 1: []}}

        for pipeline in pipelines:
            if pipeline.model is None or "sensitive_group" not in pipeline.test_data["X"].columns:
                continue
                
            try:
                X_test = pipeline.test_data["X"]
                y_test = pipeline.test_data["y"]
                y_pred = pipeline.model.predict(X_test)
                
                group0_mask = X_test["sensitive_group"] == 0
                group1_mask = X_test["sensitive_group"] == 1
                
                if pipeline.task_type == TaskType.CLASSIFICATION:
                    err0 = 1 - accuracy_score(y_test[group0_mask], y_pred[group0_mask]) if group0_mask.any() else np.nan
                    err1 = 1 - accuracy_score(y_test[group1_mask], y_pred[group1_mask]) if group1_mask.any() else np.nan
                    if not np.isnan(err0): group_errors["Classification"][0].append(err0)
                    if not np.isnan(err1): group_errors["Classification"][1].append(err1)
                else:
                    err0 = mean_absolute_error(y_test[group0_mask], y_pred[group0_mask]) if group0_mask.any() else np.nan
                    err1 = mean_absolute_error(y_test[group1_mask], y_pred[group1_mask]) if group1_mask.any() else np.nan
                    if not np.isnan(err0): group_errors["Regression"][0].append(err0)
                    if not np.isnan(err1): group_errors["Regression"][1].append(err1)
            except Exception:
                continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (task_name, errors) in enumerate(group_errors.items()):
            mean_err0 = np.mean(errors[0]) if errors[0] else 0
            mean_err1 = np.mean(errors[1]) if errors[1] else 0
            
            axes[idx].bar(["Group 0", "Group 1"], [mean_err0, mean_err1], color=["#1f77b4", "#ff7f0e"])
            axes[idx].set_title(f"Mean Error by Group ({task_name})")
            axes[idx].set_ylabel("Error Rate" if task_name == "Classification" else "Mean Absolute Error")

        plt.tight_layout()
        filepath = os.path.join(REPORTS_DIR, "error_by_group.png")
        plt.savefig(filepath)
        plt.close()

    def plot_groupwise_confusion_matrices(self, pipelines: List[PipelineInstance]):
        """Side-by-side confusion matrices for each group to show differing performance."""
        # Find one suitable pipeline to plot (plotting an aggregate CM is possible but complex)
        clf_pipelines = [p for p in pipelines if p.task_type == TaskType.CLASSIFICATION and p.model is not None and "sensitive_group" in p.test_data["X"].columns]
        if not clf_pipelines:
            return
            
        # Just pick the first one with fairness concerns
        pipeline = clf_pipelines[0]
        try:
            X_test = pipeline.test_data["X"]
            y_test = pipeline.test_data["y"]
            y_pred = pipeline.model.predict(X_test)
            
            group0_mask = X_test["sensitive_group"] == 0
            group1_mask = X_test["sensitive_group"] == 1
            
            cm0 = confusion_matrix(y_test[group0_mask], y_pred[group0_mask])
            cm1 = confusion_matrix(y_test[group1_mask], y_pred[group1_mask])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            sns.heatmap(cm0, annot=True, fmt='d', ax=axes[0], cmap='Blues')
            axes[0].set_title("Confusion Matrix (Group 0)")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True")
            
            sns.heatmap(cm1, annot=True, fmt='d', ax=axes[1], cmap='Oranges')
            axes[1].set_title("Confusion Matrix (Group 1)")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")
            
            plt.tight_layout()
            filepath = os.path.join(REPORTS_DIR, "groupwise_confusion_matrices.png")
            plt.savefig(filepath)
            plt.close()
        except Exception:
            pass
