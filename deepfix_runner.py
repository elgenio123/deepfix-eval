import json
import os
from typing import Dict, List
from dotenv import load_dotenv

from dataset_generator import PipelineInstance
from config import RAW_RESULTS_DIR

from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import TabularDataset

# Load environment variables from .env
load_dotenv()

class DeepFixRunner:
    """
    Runner for DeepFix. Uses the official DeepFix Python SDK.
    """
    
    def __init__(self):
        # API key is automatically picked up from os.environ by the SDK if DEEPFIX_API_KEY is set.
        self.client = DeepFixClient(api_url="http://localhost:8844/v2/analyse")
        
        # Ensure output directory exists
        os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    def run(self, pipeline: PipelineInstance) -> str:
        """
        Run DeepFix on a pipeline and return the raw NL report.
        """
        # Convert PipelineInstance train_data/test_data to TabularDataset format
        X_train = pipeline.train_data["X"]
        y_train = pipeline.train_data["y"]
        train_df = X_train.copy()
        train_df["target"] = y_train

        X_test = pipeline.test_data["X"]
        y_test = pipeline.test_data["y"]
        test_df = X_test.copy()
        test_df["target"] = y_test

        dataset_name = f"pipeline_{pipeline.pipeline_id}"
        label = "target"
        
        # We only have numeric features from make_classification in our current generator
        cat_features = None

        train_data = TabularDataset(
            dataset=train_df, 
            dataset_name=dataset_name, 
            label=label, 
            cat_features=cat_features
        )
        val_data = TabularDataset(
            dataset=test_df, 
            dataset_name=dataset_name, 
            label=label, 
            cat_features=cat_features
        )

        clf = pipeline.model
        model_name = clf.__class__.__name__

        try:
            result = self.client.get_diagnosis(
                train_data=train_data,
                test_data=val_data,
                model_name=model_name,
                model=clf,
                language="english",
            )
            # We assume result.to_text(verbose=False) returns the string. 
            # If it prints instead, we might need to capture stdout, but standard practice is returning the string.
            report = result.to_text(verbose=False)
            
            # If it returns None and just prints, you can capture it:
            if report is None:
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    result.to_text(verbose=False)
                report = f.getvalue()
                
        except Exception as e:
            report = f"DeepFix Execution Failed: {str(e)}"

        return report

    def run_all(self, pipelines: List[PipelineInstance]) -> Dict[str, str]:
        """
        Run DeepFix on all pipelines and save raw results.
        """
        results = {}
        for idx, pipeline in enumerate(pipelines):
            print(f"   -> Processing pipeline {idx+1}/{len(pipelines)} (ID: {pipeline.pipeline_id})")
            report = self.run(pipeline)
            results[pipeline.pipeline_id] = report
            
        # Save raw outputs
        output_file = os.path.join(RAW_RESULTS_DIR, "deepfix_raw_outputs.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
