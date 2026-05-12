"""
config.py — Central configuration for the DeepFix Evaluation Framework.

Contains the issue taxonomy, severity definitions, synonym mappings,
difficulty profiles, and all reproducibility constants.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42
NUM_REPETITIONS = 3  # For stability analysis


# ─────────────────────────────────────────────
# Severity Levels
# ─────────────────────────────────────────────
class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __str__(self) -> str:
        return self.value


# ─────────────────────────────────────────────
# Issue Categories
# ─────────────────────────────────────────────
class IssueCategory(str, Enum):
    DATA = "DATA"
    MODEL = "MODEL"
    PIPELINE = "PIPELINE"

    def __str__(self) -> str:
        return self.value


# ─────────────────────────────────────────────
# Task Types
# ─────────────────────────────────────────────
class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def __str__(self) -> str:
        return self.value


# ─────────────────────────────────────────────
# Canonical Issue Types
# ─────────────────────────────────────────────
class IssueType(str, Enum):
    # DATA issues
    MISSING_VALUES = "missing_values"
    MULTICOLLINEARITY = "multicollinearity"
    CLASS_IMBALANCE = "class_imbalance"
    DATA_LEAKAGE = "data_leakage"
    NOISY_FEATURES = "noisy_features"
    FAIRNESS_CONCERNS = "fairness_concerns"
    COVARIATE_DRIFT = "covariate_drift"

    # MODEL issues
    # UNFITTED_MODEL = "unfitted_model"
    BAD_HYPERPARAMETERS = "bad_hyperparameters"
    NO_RANDOM_STATE = "no_random_state"
    OVERFIT_UNDERFIT = "overfit_underfit"

    # PIPELINE issues
    WRONG_SPLIT = "wrong_split"
    # UNUSED_FEATURES = "unused_features"
    FEATURE_ENGINEERING_MISTAKES = "feature_engineering_mistakes"

    def __str__(self) -> str:
        return self.value


# ─────────────────────────────────────────────
# Issue Metadata Registry
# ─────────────────────────────────────────────
@dataclass
class IssueMeta:
    """Metadata for a single issue type."""
    issue_type: IssueType
    category: IssueCategory
    default_severity: Severity
    description: str


ISSUE_REGISTRY: Dict[IssueType, IssueMeta] = {
    IssueType.MISSING_VALUES: IssueMeta(
        issue_type=IssueType.MISSING_VALUES,
        category=IssueCategory.DATA,
        default_severity=Severity.MEDIUM,
        description="Dataset contains missing or null values",
    ),
    IssueType.MULTICOLLINEARITY: IssueMeta(
        issue_type=IssueType.MULTICOLLINEARITY,
        category=IssueCategory.DATA,
        default_severity=Severity.MEDIUM,
        description="Features are highly correlated with each other",
    ),
    IssueType.CLASS_IMBALANCE: IssueMeta(
        issue_type=IssueType.CLASS_IMBALANCE,
        category=IssueCategory.DATA,
        default_severity=Severity.HIGH,
        description="Target class distribution is heavily skewed",
    ),
    IssueType.DATA_LEAKAGE: IssueMeta(
        issue_type=IssueType.DATA_LEAKAGE,
        category=IssueCategory.DATA,
        default_severity=Severity.HIGH,
        description="Target or test information leaks into training features",
    ),
    IssueType.NOISY_FEATURES: IssueMeta(
        issue_type=IssueType.NOISY_FEATURES,
        category=IssueCategory.DATA,
        default_severity=Severity.LOW,
        description="Irrelevant or random noise features present",
    ),
    IssueType.FAIRNESS_CONCERNS: IssueMeta(
        issue_type=IssueType.FAIRNESS_CONCERNS,
        category=IssueCategory.DATA,
        default_severity=Severity.HIGH,
        description="Dataset or model is biased against a specific group",
    ),
    # IssueType.UNFITTED_MODEL: IssueMeta(
    #     issue_type=IssueType.UNFITTED_MODEL,
    #     category=IssueCategory.MODEL,
    #     default_severity=Severity.HIGH,
    #     description="Model was not fitted / trained before evaluation",
    # ),
    IssueType.BAD_HYPERPARAMETERS: IssueMeta(
        issue_type=IssueType.BAD_HYPERPARAMETERS,
        category=IssueCategory.MODEL,
        default_severity=Severity.MEDIUM,
        description="Model hyperparameters are set to extreme or poor values",
    ),
    IssueType.NO_RANDOM_STATE: IssueMeta(
        issue_type=IssueType.NO_RANDOM_STATE,
        category=IssueCategory.MODEL,
        default_severity=Severity.LOW,
        description="Model lacks a fixed random_state for reproducibility",
    ),
    IssueType.WRONG_SPLIT: IssueMeta(
        issue_type=IssueType.WRONG_SPLIT,
        category=IssueCategory.PIPELINE,
        default_severity=Severity.MEDIUM,
        description="Train/test split ratio is inappropriate",
    ),
    # IssueType.UNUSED_FEATURES: IssueMeta(
    #     issue_type=IssueType.UNUSED_FEATURES,
    #     category=IssueCategory.PIPELINE,
    #     default_severity=Severity.MEDIUM,
    #     description="Some features are not used during training but present in data",
    # ),
    IssueType.COVARIATE_DRIFT: IssueMeta(
        issue_type=IssueType.COVARIATE_DRIFT,
        category=IssueCategory.DATA,
        default_severity=Severity.HIGH,
        description="Feature distributions shift between training and test data",
    ),
    IssueType.FEATURE_ENGINEERING_MISTAKES: IssueMeta(
        issue_type=IssueType.FEATURE_ENGINEERING_MISTAKES,
        category=IssueCategory.PIPELINE,
        default_severity=Severity.MEDIUM,
        description="Categorical variables are misencoded or feature engineering leaks info",
    ),
    IssueType.OVERFIT_UNDERFIT: IssueMeta(
        issue_type=IssueType.OVERFIT_UNDERFIT,
        category=IssueCategory.MODEL,
        default_severity=Severity.HIGH,
        description="Model overfits or underfits the training data",
    ),
}


# ─────────────────────────────────────────────
# Difficulty Profiles
# ─────────────────────────────────────────────
class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    def __str__(self) -> str:
        return self.value


DIFFICULTY_PROFILES = {
    Difficulty.EASY: {
        "num_issues": (1, 1),       # exactly 1 issue
        "description": "Single isolated issue",
    },
    Difficulty.MEDIUM: {
        "num_issues": (2, 3),       # 2–3 issues
        "description": "Multiple issues from different categories",
    },
    Difficulty.HARD: {
        "num_issues": (4, 6),       # 4–6 interacting issues
        "description": "Multiple interacting issues across categories",
    },
}


# ─────────────────────────────────────────────
# Dataset Generation
# ─────────────────────────────────────────────
# These counts are applied to EACH task type (classification AND regression).
# e.g. sum = 10 means 10 classification + 10 regression = 20 total API requests.
PIPELINE_COUNTS_PER_TASK = {
    Difficulty.EASY: 0,
    Difficulty.MEDIUM: 1,
    Difficulty.HARD: 0,
}
TOTAL_PIPELINES = sum(PIPELINE_COUNTS_PER_TASK.values()) * len(TaskType)

# Synthetic dataset parameters
# DEFAULT_N_SAMPLES = 1000
# DEFAULT_N_FEATURES = 15
# DEFAULT_N_INFORMATIVE = 10
# DEFAULT_N_CLASSES = 3
DEFAULT_TEST_SIZE = 0.2


# ─────────────────────────────────────────────
# Injection Parameters
# ─────────────────────────────────────────────
INJECTION_PARAMS = {
    IssueType.MISSING_VALUES: {
        "fraction_range": (0.05, 0.50),  # 5–50% of cells become NaN
    },
    IssueType.MULTICOLLINEARITY: {
        "noise_scale": 0.01,  # near-perfect linear combination
    },
    IssueType.CLASS_IMBALANCE: {
        "minority_ratio": 0.05,  # reduce minority to 5% of majority
    },
    IssueType.DATA_LEAKAGE: {
        "noise_scale": 0.1,  # slight noise on leaked target
    },
    IssueType.NOISY_FEATURES: {
        "num_noisy": 5,  # number of random columns to add
    },
    IssueType.FAIRNESS_CONCERNS: {
        "sensitive_feature": "sensitive_group",
        "bias_factor": 0.5,
    },
    # IssueType.UNFITTED_MODEL: {},  # no params needed
    IssueType.BAD_HYPERPARAMETERS: {
        "max_depth": 1,
        "learning_rate": 10.0,
    },
    IssueType.NO_RANDOM_STATE: {},  # no params needed
    IssueType.WRONG_SPLIT: {
        "split_ratio": 0.99,  # 99% train / 1% test
    },
    # IssueType.UNUSED_FEATURES: {
    #     "drop_fraction": 0.5,  # drop 50% of features during training
    # },
    IssueType.COVARIATE_DRIFT: {
        "shift_mean": 3.0,      # add to test features
        "shift_scale": 1.5,     # multiply noise on test features
    },
    IssueType.FEATURE_ENGINEERING_MISTAKES: {
        "num_categories": 5,    # raw integer-encoded categorical
    },
    IssueType.OVERFIT_UNDERFIT: {
        "overfit": {"max_iter": 1000, "min_samples_leaf": 1, "max_depth": None},
        "underfit": {"max_iter": 5, "max_depth": 1},
    },
}


# ─────────────────────────────────────────────
# Evaluation Split
# ─────────────────────────────────────────────
VALIDATION_FRACTION = 0.30  # 30% for parser tuning
TEST_FRACTION = 0.70         # 70% for final evaluation


# ─────────────────────────────────────────────
# Output / Logging
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
REPORTS_DIR = "outputs/reports"
RAW_RESULTS_DIR = "outputs/raw"
LOG_FILE = "outputs/experiment.log"
