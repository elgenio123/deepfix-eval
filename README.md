# DeepFix Evaluation Framework

A comprehensive benchmarking suite designed to evaluate the performance of the **DeepFix** ML diagnostic tool. This framework injects controlled issues (data, model, and pipeline errors) into machine learning pipelines and measures DeepFix's ability to detect and categorize them.

## 🚀 Overview

The framework automates the end-to-end evaluation process:
1.  **Dataset Generation**: Creates synthetic pipelines using `seaborn` datasets (iris, diamonds) with precisely injected issues like missing values, multicollinearity, and covariate drift.
2.  **Execution**: Orchestrates the `deepfix_sdk` to analyze these pipelines.
3.  **Parsing**: Uses LLMs (via OpenRouter) to parse unstructured natural language reports from DeepFix into structured predictions.
4.  **Evaluation**: Computes rigorous metrics including Precision, Recall, F1 (Macro/Micro), and Pipeline-level Detection Rates.
5.  **Reporting**: Generates detailed Markdown reports and visualizations (residual plots, error-by-group, etc.).

## 🛠️ Project Structure

- `main.py`: Entry point for standard end-to-end experiments.
- `comparative_eval.py`: Specialized script to compare different DeepFix versions (e.g., Online vs. Local).
- `config.py`: Central configuration for issue taxonomy, severity levels, and experiment parameters.
- `dataset_generator.py`: The engine for generating pipelines and injecting issues.
- `deepfix_runner.py`: Wrapper for the DeepFix SDK.
- `output_parser.py`: LLM-based parser for structured data extraction.
- `evaluator.py`: Logic for computing evaluation metrics.
- `reporter.py`: Generates Markdown and JSON result files.
- `visualizer.py`: Creates performance plots and charts.

## ⚙️ Setup

### 1. Prerequisites
- Python 3.8+
- A valid `DEEPFIX_API_KEY`.
- An `OPENROUTER_API_KEY` for report parsing.

### 2. Installation
Clone the repository and set up the virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:

```env
DEEPFIX_API_KEY=your_deepfix_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
# Optional:
OPENROUTER_MODEL=google/openai/gpt-oss-120b:free
```

## 🏃 Running the Framework

### Standard Experiment
To run a single experiment using the settings in `config.py`:
```bash
python main.py
```

### Comparative Evaluation
To compare the official Online DeepFix against a Local instance:
```bash
python comparative_eval.py
```
*Note: Ensure your local DeepFix instance is running at the URL specified in `comparative_eval.py` (default: `http://localhost:8844`).*

## 📊 Outputs

- **Standard Runs**: Results are saved in `outputs/`.
- **Comparative Runs**: Results are organized in `outputs_comparative/`:
    - `online/`: Reports and raw data for the online version.
    - `local/`: Reports and raw data for the local version.
    - `reports/`: The final comparative report (`comparative_report.md`).

## 🧪 Issue Taxonomy

The framework currently supports the following issues:
- **DATA**: `missing_values`, `multicollinearity`, `class_imbalance`, `data_leakage`, `noisy_features`, `fairness_concerns`, `covariate_drift`.
- **MODEL**: `bad_hyperparameters`, `no_random_state`, `overfit_underfit`.
- **PIPELINE**: `wrong_split`, `feature_engineering_mistakes`.

## 📜 License
This project is for internal evaluation and development purposes.
