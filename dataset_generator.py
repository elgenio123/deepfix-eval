import os
import uuid
import random
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Set
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.datasets import fetch_openml

import config
from config import (
    IssueType,
    TaskType,
    Difficulty,
    DIFFICULTY_PROFILES,
    INJECTION_PARAMS,
    RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    ISSUE_REGISTRY
)

class PipelineInstance:
    """Represents a single generated ML pipeline with recorded inherent and injected issues."""
    def __init__(self, pipeline_id: str, difficulty: Difficulty, task_type: TaskType):
        self.pipeline_id = pipeline_id
        self.difficulty = difficulty
        self.task_type = task_type
        self.train_data: Dict[str, Any] = {}
        self.test_data: Dict[str, Any] = {}
        self.model: Any = None
        self.ground_truth_issues: List[Dict[str, Any]] = []

class DatasetGenerator:
    """Generates ML pipelines using high-challenge benchmarks, recording inherent issues as ground truth."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        os.makedirs(config.DATA_DIR, exist_ok=True)

    def _generate_base_data(self, task_type: TaskType) -> Tuple[pd.DataFrame, pd.Series, List[IssueType]]:
        """Fetch real-world datasets and identify their inherent issues."""
        inherent_issues = []
        
        if task_type == TaskType.CLASSIFICATION:
            # BAF (Bank Account Fraud) Dataset - Local Base.csv
            # Source: Bank Account Fraud Dataset Suite (NeurIPS 2022)
            try:
                base_dir = os.path.join(config.DATA_DIR, "base_classification")
                base_path = os.path.join(base_dir, "Base.csv")
                
                print(f"   -> Loading BAF dataset from {base_path}...")
                if not os.path.exists(base_path):
                    raise FileNotFoundError(f"Base dataset not found at {base_path}")
                
                df = pd.read_csv(base_path)
                
                # Target is 'fraud_bool'
                if 'fraud_bool' not in df.columns:
                    raise ValueError("Target column 'fraud_bool' not found in dataset")
                
                y = df['fraud_bool'].astype(int)
                X = df.drop(columns=['fraud_bool'])

                # Inherent Issues in BAF:
                inherent_issues.extend([
                    IssueType.CLASS_IMBALANCE,  # Fraud is highly imbalanced (~1%)
                    IssueType.FEATURE_ENGINEERING_MISTAKES, # Contains raw categorical features
                    IssueType.DATA_LEAKAGE # Month/Date columns can sometimes cause leakage if not handled
                ])
                
                print(f"      -> BAF classification data loaded successfully")

                if len(X) > 5000:
                    X = X.sample(n=5000, random_state=self.seed)
                    y = y.loc[X.index]
                
                # Encode categories for the base model
                for col in X.select_dtypes(['object', 'category']).columns:
                    X[col] = X[col].astype('category').cat.codes
                
                # Also encode any columns that are clearly categorical but read as object
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category').cat.codes
                    
            except Exception as e:
                print(f"   -> Fallback to synthetic classification: {e}")
                from sklearn.datasets import make_classification
                X_arr, y_arr = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=self.seed)
                # Use slightly more descriptive names for synthetic fallback
                cols = [f"feature_{i}" for i in range(20)]
                if len(cols) > 2:
                    cols[0] = "age_group"
                    cols[1] = "transaction_amount"
                X = pd.DataFrame(X_arr, columns=cols)
                y = pd.Series(y_arr, name="target")
        else:
            # Lending Club Loan Data - Local lending_club.csv or OpenML fallback
            try:
                base_dir = os.path.join(config.DATA_DIR, "base_regression")
                base_path = os.path.join(base_dir, "lending_club.csv")
                
                if os.path.exists(base_path):
                    print(f"   -> Loading Lending Club dataset from {base_path}...")
                    df = pd.read_csv(base_path)
                    
                    # Target is 'target' (saved by previous runs) or 'int.rate'/'int_rate'
                    target_col = 'target' if 'target' in df.columns else ('int.rate' if 'int.rate' in df.columns else ('int_rate' if 'int_rate' in df.columns else None))
                    
                    if target_col:
                        y = df[target_col]
                        X = df.drop(columns=[target_col])
                    else:
                        raise ValueError("Target variable not found in local CSV")
                else:
                    print("   -> Fetching 'Lending Club' (43729) from OpenML...")
                    data = fetch_openml(data_id=43729, as_frame=True, parser='auto', data_home=config.DATA_DIR)
                    df = data.frame
                    
                    # Correct target extraction for 43729
                    target_col = 'int.rate' if 'int.rate' in df.columns else ('int_rate' if 'int_rate' in df.columns else None)
                    if target_col:
                        y = df[target_col]
                        X = df.drop(columns=[target_col])
                    else:
                        y, X = data.target, data.data

                # Inherent Issues in Lending Club:
                inherent_issues.extend([
                    IssueType.MISSING_VALUES,
                    IssueType.MULTICOLLINEARITY,
                    IssueType.FEATURE_ENGINEERING_MISTAKES
                ])
                
                if y is None:
                    raise ValueError("Target variable not found")
                
                y = y.astype(float)

                # Save if it was fetched
                if not os.path.exists(base_path):
                    os.makedirs(base_dir, exist_ok=True)
                    df_base = X.copy()
                    df_base['target'] = y
                    df_base.to_csv(base_path, index=False)
                    print(f"      -> Base regression data saved to {base_dir}")
                
                if len(X) > 5000:
                    X = X.sample(n=5000, random_state=self.seed)
                    y = y.loc[X.index]
                
                for col in X.select_dtypes(['object', 'category']).columns:
                    X[col] = X[col].astype('category').cat.codes
                
                # Also encode any columns that are clearly categorical but read as object
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category').cat.codes
                    
            except Exception as e:
                print(f"   -> Fallback to synthetic regression: {e}")
                from sklearn.datasets import make_regression
                X_arr, y_arr = make_regression(n_samples=2000, n_features=20, n_informative=10, noise=0.1, random_state=self.seed)
                # Use slightly more descriptive names for synthetic fallback
                cols = [f"feature_{i}" for i in range(20)]
                if len(cols) > 2:
                    cols[0] = "income"
                    cols[1] = "credit_score"
                X = pd.DataFrame(X_arr, columns=cols)
                y = pd.Series(y_arr, name="target")
            
        return X.reset_index(drop=True), y.reset_index(drop=True), inherent_issues

    def _inject_data_issues(self, X: pd.DataFrame, y: pd.Series, issues_to_inject: List[IssueType], task_type: TaskType) -> Tuple[pd.DataFrame, pd.Series]:
        X_inj = X.copy()
        y_inj = y.copy()

        # Inject only if NOT already inherent (unless we want to amplify them)
        if IssueType.MISSING_VALUES in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.MISSING_VALUES]
            frac = self.rng.uniform(*params["fraction_range"])
            mask = self.rng.random(X_inj.shape) < frac
            X_inj = X_inj.mask(mask)

        if IssueType.MULTICOLLINEARITY in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.MULTICOLLINEARITY]
            col_to_copy = X_inj.columns[0]
            X_inj["collinear_feature"] = X_inj[col_to_copy] + self.rng.normal(scale=params["noise_scale"], size=len(X_inj))

        if IssueType.CLASS_IMBALANCE in issues_to_inject:
            # We amplify imbalance if requested
            minority_class = 1 
            minority_idx = y_inj[y_inj == minority_class].index
            other_idx = y_inj[y_inj != minority_class].index
            n_keep = max(1, int(y_inj.value_counts().max() * INJECTION_PARAMS[IssueType.CLASS_IMBALANCE]["minority_ratio"]))
            keep_minority_idx = self.rng.choice(minority_idx, size=min(n_keep, len(minority_idx)), replace=False)
            keep_idx = np.concatenate([other_idx, keep_minority_idx])
            X_inj = X_inj.loc[keep_idx].reset_index(drop=True)
            y_inj = y_inj.loc[keep_idx].reset_index(drop=True)

        if IssueType.DATA_LEAKAGE in issues_to_inject:
            X_inj["leaked_target"] = y_inj + self.rng.normal(scale=INJECTION_PARAMS[IssueType.DATA_LEAKAGE]["noise_scale"], size=len(y_inj))

        if IssueType.NOISY_FEATURES in issues_to_inject:
            for i in range(INJECTION_PARAMS[IssueType.NOISY_FEATURES]["num_noisy"]):
                X_inj[f"noisy_{i}"] = self.rng.normal(size=len(X_inj))

        if IssueType.FAIRNESS_CONCERNS in issues_to_inject:
            group_name = INJECTION_PARAMS[IssueType.FAIRNESS_CONCERNS]["sensitive_feature"]
            if group_name not in X_inj.columns:
                X_inj[group_name] = self.rng.choice([0, 1], size=len(X_inj))
            
            group0_mask = X_inj[group_name] == 0
            bias = INJECTION_PARAMS[IssueType.FAIRNESS_CONCERNS]["bias_factor"]
            if task_type == TaskType.CLASSIFICATION:
                flip_mask = group0_mask & (y_inj == 1) & (self.rng.random(len(X_inj)) < bias)
                y_inj.loc[flip_mask] = 0
            else:
                y_inj.loc[group0_mask] -= bias * y_inj.std()

        if IssueType.FEATURE_ENGINEERING_MISTAKES in issues_to_inject:
            X_inj["misencoded_cat"] = self.rng.integers(0, INJECTION_PARAMS[IssueType.FEATURE_ENGINEERING_MISTAKES]["num_categories"], size=len(X_inj))
            X_inj["future_leak"] = y_inj.values * 0.8 + self.rng.normal(scale=0.2, size=len(X_inj))

        return X_inj, y_inj

    def generate_pipeline(self, difficulty: Difficulty, task_type: TaskType) -> PipelineInstance:
        """Generate a pipeline, recording inherent issues first, then supplementing with injected ones."""
        pipeline = PipelineInstance(str(uuid.uuid4()), difficulty, task_type)
        
        # 1. Fetch data and identify INHERENT issues
        X, y, inherent_issues = self._generate_base_data(task_type)
        
        # Record inherent issues in ground truth immediately
        for issue in inherent_issues:
            meta = ISSUE_REGISTRY[issue]
            pipeline.ground_truth_issues.append({
                "type": issue.value, "severity": meta.default_severity.value, "category": meta.category.value, "source": "inherent"
            })
            
        # 2. Determine how many SUPPLEMENTARY issues to inject
        profile = DIFFICULTY_PROFILES[difficulty]
        target_num = self.rng.integers(profile["num_issues"][0], profile["num_issues"][1] + 1)
        
        # We try to reach the target_num. If inherent already exceeds it, we might still add 1-2 "new" types.
        num_to_inject = max(0, target_num - len(inherent_issues))
        if num_to_inject == 0 and target_num > 0:
            num_to_inject = 1 # Always inject at least one supplementary for variety
            
        available_types = [t for t in IssueType if t not in inherent_issues]
        if task_type == TaskType.REGRESSION and IssueType.CLASS_IMBALANCE in available_types:
            available_types.remove(IssueType.CLASS_IMBALANCE)
            
        to_inject = self.rng.choice(available_types, size=min(num_to_inject, len(available_types)), replace=False)
        to_inject = [IssueType(i) for i in to_inject]
        
        # Record injected issues
        for issue in to_inject:
            meta = ISSUE_REGISTRY[issue]
            pipeline.ground_truth_issues.append({
                "type": issue.value, "severity": meta.default_severity.value, "category": meta.category.value, "source": "injected"
            })
            
        # 3. Apply injection
        X, y = self._inject_data_issues(X, y, to_inject, task_type)
        
        # 4. Pipeline/Model setup
        test_size = DEFAULT_TEST_SIZE
        if IssueType.WRONG_SPLIT in to_inject or IssueType.WRONG_SPLIT in inherent_issues:
            test_size = 1.0 - INJECTION_PARAMS[IssueType.WRONG_SPLIT]["split_ratio"]
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.seed)
        pipeline.train_data, pipeline.test_data = {"X": X_train, "y": y_train}, {"X": X_test, "y": y_test}
        
        # Save datasets locally
        try:
            instance_dir = os.path.join(config.DATA_DIR, pipeline.pipeline_id)
            os.makedirs(instance_dir, exist_ok=True)
            
            train_df = X_train.copy()
            train_df['target'] = y_train
            train_df.to_csv(os.path.join(instance_dir, "train.csv"), index=False)
            
            test_df = X_test.copy()
            test_df['target'] = y_test
            test_df.to_csv(os.path.join(instance_dir, "test.csv"), index=False)
            
            # Also save ground truth issues for reference
            with open(os.path.join(instance_dir, "ground_truth.json"), "w") as f:
                json.dump(pipeline.ground_truth_issues, f, indent=4)
                
            print(f"      -> Pipeline {pipeline.pipeline_id} data saved to {instance_dir}")
        except Exception as e:
            print(f"      -> Warning: Could not save data for pipeline {pipeline.pipeline_id}: {e}")

        # Covariate drift
        if IssueType.COVARIATE_DRIFT in to_inject:
            p = INJECTION_PARAMS[IssueType.COVARIATE_DRIFT]
            num_cols = X_test.select_dtypes(include=[np.number]).columns
            noise = self.rng.normal(loc=p["shift_mean"], scale=p["shift_scale"], size=(len(X_test), len(num_cols)))
            X_test_shifted = X_test.copy()
            X_test_shifted[num_cols] += noise
            pipeline.test_data["X"] = X_test_shifted
        
        # Model training
        model_kwargs = {"random_state": self.seed} if IssueType.NO_RANDOM_STATE not in (to_inject + inherent_issues) else {}
        if IssueType.BAD_HYPERPARAMETERS in to_inject:
            model_kwargs.update(INJECTION_PARAMS[IssueType.BAD_HYPERPARAMETERS])
        if IssueType.OVERFIT_UNDERFIT in to_inject:
            model_kwargs.update(INJECTION_PARAMS[IssueType.OVERFIT_UNDERFIT][self.rng.choice(["overfit", "underfit"])])
            
        model = HistGradientBoostingClassifier(**model_kwargs) if task_type == TaskType.CLASSIFICATION else HistGradientBoostingRegressor(**model_kwargs)
        try: model.fit(X_train, y_train)
        except Exception: pass
        
        pipeline.model = model
        return pipeline

    def generate_dataset(self, counts_per_task: Dict[Difficulty, int]) -> List[PipelineInstance]:
        """Generate a complete benchmark dataset."""
        dataset = []
        for task_type in TaskType:
            for diff, count in counts_per_task.items():
                for _ in range(count):
                    dataset.append(self.generate_pipeline(diff, task_type))
        return dataset
