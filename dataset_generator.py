import uuid
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from config import (
    IssueType,
    TaskType,
    Difficulty,
    DIFFICULTY_PROFILES,
    INJECTION_PARAMS,
    DEFAULT_N_SAMPLES,
    DEFAULT_N_FEATURES,
    DEFAULT_N_INFORMATIVE,
    DEFAULT_N_CLASSES,
    DEFAULT_TEST_SIZE,
    RANDOM_SEED
)

class PipelineInstance:
    """Represents a single generated ML pipeline with injected issues."""
    def __init__(self, pipeline_id: str, difficulty: Difficulty, task_type: TaskType):
        self.pipeline_id = pipeline_id
        self.difficulty = difficulty
        self.task_type = task_type
        self.train_data: Dict[str, Any] = {}
        self.test_data: Dict[str, Any] = {}
        self.model: Any = None
        self.ground_truth_issues: List[Dict[str, Any]] = []

    def to_dict(self):
        return {
            "pipeline_id": self.pipeline_id,
            "difficulty": str(self.difficulty),
            "task_type": str(self.task_type),
            "ground_truth_issues": self.ground_truth_issues,
            "has_model": self.model is not None,
            "train_shape": self.train_data['X'].shape if 'X' in self.train_data else None,
            "test_shape": self.test_data['X'].shape if 'X' in self.test_data else None,
        }

class DatasetGenerator:
    """Generates synthetic ML pipelines with controlled error injection."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def _generate_base_data(self, task_type: TaskType) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate clean, base data."""
        if task_type == TaskType.CLASSIFICATION:
            df = sns.load_dataset("iris")
            # if len(df) > DEFAULT_N_SAMPLES:
            #     df = df.sample(n=DEFAULT_N_SAMPLES, random_state=self.seed)
            X = df.drop(columns=["species"])
            y = df["species"].astype("category").cat.codes
            y.name = "target"
        else:
            df = sns.load_dataset("diamonds")
            # if len(df) > DEFAULT_N_SAMPLES:
            #     df = df.sample(n=DEFAULT_N_SAMPLES, random_state=self.seed)
            X = df.drop(columns=["price"])
            for col in X.select_dtypes(['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
            y = df["price"]
            y.name = "target"
            
        return X.reset_index(drop=True), y.reset_index(drop=True)

    def _inject_data_issues(self, X: pd.DataFrame, y: pd.Series, issues_to_inject: List[IssueType], task_type: TaskType) -> Tuple[pd.DataFrame, pd.Series]:
        X_inj = X.copy()
        y_inj = y.copy()

        if IssueType.MISSING_VALUES in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.MISSING_VALUES]
            frac = self.rng.uniform(*params["fraction_range"])
            mask = self.rng.random(X_inj.shape) < frac
            X_inj = X_inj.mask(mask)

        if IssueType.MULTICOLLINEARITY in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.MULTICOLLINEARITY]
            col_to_copy = X_inj.columns[0]
            noise = self.rng.normal(scale=params["noise_scale"], size=len(X_inj))
            X_inj["collinear_feature"] = X_inj[col_to_copy] + noise

        if IssueType.CLASS_IMBALANCE in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.CLASS_IMBALANCE]
            minority_class = self.rng.choice(y_inj.unique())
            
            minority_idx = y_inj[y_inj == minority_class].index
            other_idx = y_inj[y_inj != minority_class].index
            
            largest_class_size = y_inj.value_counts().max()
            n_minority_keep = int(largest_class_size * params["minority_ratio"])
            n_minority_keep = max(1, min(n_minority_keep, len(minority_idx)))
            
            keep_minority_idx = self.rng.choice(minority_idx, size=n_minority_keep, replace=False)
            
            keep_idx = np.concatenate([other_idx, keep_minority_idx])
            X_inj = X_inj.loc[keep_idx].reset_index(drop=True)
            y_inj = y_inj.loc[keep_idx].reset_index(drop=True)

        if IssueType.DATA_LEAKAGE in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.DATA_LEAKAGE]
            noise = self.rng.normal(scale=params["noise_scale"], size=len(y_inj))
            X_inj["leaked_target"] = y_inj + noise

        if IssueType.NOISY_FEATURES in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.NOISY_FEATURES]
            for i in range(params["num_noisy"]):
                X_inj[f"noisy_{i}"] = self.rng.normal(size=len(X_inj))

        if IssueType.FAIRNESS_CONCERNS in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.FAIRNESS_CONCERNS]
            group_name = params["sensitive_feature"]
            X_inj[group_name] = self.rng.choice([0, 1], size=len(X_inj))
            
            group0_mask = X_inj[group_name] == 0
            if task_type == TaskType.CLASSIFICATION:
                pos_mask = y_inj == 1
                flip_mask = group0_mask & pos_mask & (self.rng.random(len(X_inj)) < params["bias_factor"])
                y_inj.loc[flip_mask] = 0
            else:
                y_inj.loc[group0_mask] -= params["bias_factor"] * y_inj.std()

        if IssueType.FEATURE_ENGINEERING_MISTAKES in issues_to_inject:
            params = INJECTION_PARAMS[IssueType.FEATURE_ENGINEERING_MISTAKES]
            # Add a raw integer-encoded categorical (should be one-hot but isn't)
            X_inj["misencoded_cat"] = self.rng.integers(0, params["num_categories"], size=len(X_inj))
            # Add a column that leaks future info (future value correlated with target)
            X_inj["future_leak"] = y_inj.values * 0.8 + self.rng.normal(scale=0.2, size=len(X_inj))

        return X_inj, y_inj

    def generate_pipeline(self, difficulty: Difficulty, task_type: TaskType) -> PipelineInstance:
        """Generate a single pipeline with injected issues based on difficulty."""
        pipeline = PipelineInstance(
            pipeline_id=str(uuid.uuid4()),
            difficulty=difficulty,
            task_type=task_type
        )
        
        profile = DIFFICULTY_PROFILES[difficulty]
        num_issues = self.rng.integers(profile["num_issues"][0], profile["num_issues"][1] + 1)
        
        # Select random issues
        available_issues = list(IssueType)
        if task_type == TaskType.REGRESSION:
            if IssueType.CLASS_IMBALANCE in available_issues:
                available_issues.remove(IssueType.CLASS_IMBALANCE)
                
        selected_issues = self.rng.choice(available_issues, size=num_issues, replace=False)
        selected_issues = [IssueType(i) for i in selected_issues]
        
        # Record ground truth
        from config import ISSUE_REGISTRY
        for issue in selected_issues:
            meta = ISSUE_REGISTRY[issue]
            pipeline.ground_truth_issues.append({
                "type": issue.value,
                "severity": meta.default_severity.value,
                "category": meta.category.value
            })
            
        # 1. Base Data Generation
        X, y = self._generate_base_data(task_type)
        
        # 2. Inject Data Issues
        X, y = self._inject_data_issues(X, y, selected_issues, task_type)
        
        # 3. Pipeline Issues (Split)
        test_size = DEFAULT_TEST_SIZE
        if IssueType.WRONG_SPLIT in selected_issues:
            test_size = 1.0 - INJECTION_PARAMS[IssueType.WRONG_SPLIT]["split_ratio"]
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )
        
        # 4. Pipeline Issues (Unused Features)
        if IssueType.UNUSED_FEATURES in selected_issues:
            drop_frac = INJECTION_PARAMS[IssueType.UNUSED_FEATURES]["drop_fraction"]
            n_drop = int(X_train.shape[1] * drop_frac)
            if n_drop > 0:
                cols_to_drop = self.rng.choice(X_train.columns, size=n_drop, replace=False)
                X_train = X_train.drop(columns=cols_to_drop)
                # Keep in test to simulate the issue
        
        pipeline.train_data = {"X": X_train, "y": y_train}
        pipeline.test_data = {"X": X_test, "y": y_test}
        
        # 4b. Covariate Drift (shift test set distributions)
        if IssueType.COVARIATE_DRIFT in selected_issues:
            params = INJECTION_PARAMS[IssueType.COVARIATE_DRIFT]
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            noise = self.rng.normal(
                loc=params["shift_mean"],
                scale=params["shift_scale"],
                size=(len(X_test), len(numeric_cols))
            )
            X_test_shifted = X_test.copy()
            X_test_shifted[numeric_cols] = X_test_shifted[numeric_cols] + noise
            pipeline.test_data = {"X": X_test_shifted, "y": y_test}
        
        # 5. Model Issues
        model_kwargs = {}
        if IssueType.NO_RANDOM_STATE not in selected_issues:
            model_kwargs["random_state"] = self.seed
            
        if IssueType.BAD_HYPERPARAMETERS in selected_issues:
            params = INJECTION_PARAMS[IssueType.BAD_HYPERPARAMETERS]
            model_kwargs.update(params)
        
        # OVERFIT_UNDERFIT takes precedence over BAD_HYPERPARAMETERS if both selected
        if IssueType.OVERFIT_UNDERFIT in selected_issues:
            params = INJECTION_PARAMS[IssueType.OVERFIT_UNDERFIT]
            # Randomly pick overfit or underfit
            mode = self.rng.choice(["overfit", "underfit"])
            model_kwargs.update(params[mode])
            
        if task_type == TaskType.CLASSIFICATION:
            model = HistGradientBoostingClassifier(**model_kwargs)
        else:
            model = HistGradientBoostingRegressor(**model_kwargs)
        
        if IssueType.UNFITTED_MODEL not in selected_issues:
            # Handle missing values properly for HistGradientBoostingClassifier 
            # Or fill them if we want to ensure fit works (HistGradientBoostingClassifier handles NaNs natively)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                # If fitting fails due to an injected issue (like NaN, though HistGB handles them, or zero columns), 
                # we just leave it unfitted or handled gracefully.
                pass
                
        pipeline.model = model
        
        return pipeline

    def generate_dataset(self, counts_per_task: Dict[Difficulty, int]) -> List[PipelineInstance]:
        """Generate a complete benchmark dataset.
        
        Args:
            counts_per_task: Difficulty -> count mapping applied to EACH task type.
                e.g. {EASY: 5, MEDIUM: 10, HARD: 5} generates 20 classification + 20 regression = 40 pipelines.
        """
        dataset = []
        for task_type in TaskType:
            for diff, count in counts_per_task.items():
                for _ in range(count):
                    dataset.append(self.generate_pipeline(diff, task_type))
        return dataset

if __name__ == "__main__":
    from config import PIPELINE_COUNTS_PER_TASK
    gen = DatasetGenerator()
    data = gen.generate_dataset(PIPELINE_COUNTS_PER_TASK)
    print(f"Generated {len(data)} pipelines.")
    print(f"Sample ground truth: {data[0].ground_truth_issues}")
