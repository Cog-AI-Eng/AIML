# Guide 4: Add a Quality Gate with ConditionStep

A pipeline that registers every trained model -- regardless of quality -- is not production-ready. In this guide, you will add an evaluation step that computes metrics on a test set, and a ConditionStep that only registers the model if the metrics meet a threshold. This is the automated equivalent of the manual approval workflow you performed in Module 3.

---

## Steps

### Step 1 -- Create the Evaluation Script

Return to your Studio notebook and create an evaluation script:

```python
%%writefile evaluate.py
import os
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score
import joblib
import tarfile

# Paths
model_dir = "/opt/ml/processing/model"
test_dir = "/opt/ml/processing/test"
output_dir = "/opt/ml/processing/evaluation"
os.makedirs(output_dir, exist_ok=True)

# Extract model from model.tar.gz
model_tar = os.path.join(model_dir, "model.tar.gz")
with tarfile.open(model_tar, "r:gz") as tar:
    tar.extractall(path=model_dir)

model = joblib.load(os.path.join(model_dir, "model.pkl"))

# Load test data
test_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
test_df = pd.concat([pd.read_csv(os.path.join(test_dir, f)) for f in test_files])

X_test = test_df.drop(columns=["is_fraud"])
y_test = test_df["is_fraud"]

# Generate predictions and compute metrics
predictions = model.predict(X_test)

metrics = {
    "f1": float(f1_score(y_test, predictions, average="binary", zero_division=0)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, average="binary", zero_division=0)),
}

# Write metrics to JSON (SageMaker reads this via PropertyFile)
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
```

### Step 2 -- Add the Evaluation Step to the Pipeline

```python
from sagemaker.workflow.properties import PropertyFile

# Define a PropertyFile so the ConditionStep can read the metrics
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="metrics.json",
)

evaluate_step = ProcessingStep(
    name="EvaluateModel",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=process_step.properties.ProcessingOutputConfig
                .Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/pipeline/evaluation/",
        ),
    ],
    code="evaluate.py",
    property_files=[evaluation_report],
)
```

### Step 3 -- Add the ConditionStep

```python
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# Define the condition: F1 score must be >= 0.85
f1_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=evaluate_step.name,
        property_file=evaluation_report,
        json_path="f1",
    ),
    right=0.85,
)

# ConditionStep: register only if condition is met
condition_step = ConditionStep(
    name="CheckModelQuality",
    conditions=[f1_condition],
    if_steps=[register_step],    # Register if F1 >= 0.85
    else_steps=[],               # Do nothing if F1 < 0.85
)
```

### Step 4 -- Update and Re-Run the Pipeline

```python
# Update the pipeline with the new steps
pipeline_v2 = Pipeline(
    name="fraudshield-detection-pipeline",
    parameters=[instance_type, n_estimators],
    steps=[process_step, train_step, evaluate_step, condition_step],
)

# Upsert the updated definition
pipeline_v2.upsert(role_arn=role)
print("Pipeline v2 upserted")

# Start a new execution
execution_v2 = pipeline_v2.start()
print(f"Pipeline v2 execution: {execution_v2.describe()['PipelineExecutionArn']}")
```

### Step 5 -- Monitor the Updated Pipeline in the Console

1. Return to the **SageMaker console** -> **Pipelines** -> `fraudshield-detection-pipeline`.
2. Click on the **Graph** tab. The DAG should now show the updated structure:

```
PreprocessData -> TrainModel -> EvaluateModel -> CheckModelQuality
                                                       │
                                                  ┌────┴────┐
                                                  ▼         ▼
                                            RegisterModel  (skip)
```

3. Click on the **Executions** tab and find the latest execution.
4. Watch the steps progress through the DAG.

### Step 6 -- Observe the Condition Outcome

1. When the execution reaches **CheckModelQuality**, observe:
   - If the F1 score is >= 0.85, the step will branch to **RegisterModel** (green path)
   - If the F1 score is < 0.85, the step will skip registration (the else path)
2. Click on the **CheckModelQuality** step to see:
   - The condition that was evaluated
   - The actual metric value
   - Which branch was taken

### Step 7 -- Verify the Result

If the model passed the quality gate:
1. Go to **Governance** -> **Model registry** -> `fraud-detection-rf`.
2. A new version should be registered with `PendingManualApproval` status.

If the model did NOT pass:
1. No new version will appear in the registry.
2. The pipeline execution will show **Succeeded** (the pipeline itself succeeded -- the condition correctly prevented registration of a low-quality model).

---

## Presentation Checkpoint

Be prepared to show:
- The **updated DAG** with four steps including the branching ConditionStep
- The **evaluation metrics** (click on the EvaluateModel step to see the metrics JSON)
- The **CheckModelQuality** step showing which branch was taken and why
- Whether a new model version was registered (or correctly skipped)
- Explain: What does `JsonGet` do? (It reads a specific value from a JSON file produced by a previous step. Here it reads the `f1` key from `metrics.json` produced by the evaluation step.)
- Explain: What happens to the pipeline if the model fails the quality gate? (The pipeline execution still succeeds -- it just skips the registration step. A "failed condition" is not a "failed pipeline." The pipeline correctly enforced the quality gate.)
- Explain: How is this different from the manual approval in Module 3? (Manual approval is a human reviewing metrics after registration. The ConditionStep is automated -- it prevents registration entirely if metrics are below threshold. In production, you might use both: automated pre-screening + human final approval.)

---

## Key Concepts

- **PropertyFile:** A mechanism for passing structured data (JSON) between pipeline steps. The evaluation step writes metrics to a JSON file, and the ConditionStep reads from it.
- **ConditionStep:** A pipeline step that evaluates a boolean condition and branches the DAG accordingly. Supports `if_steps` (run if true) and `else_steps` (run if false).
- **`JsonGet`:** A function that extracts a value from a PropertyFile by JSON path. This is how the ConditionStep reads the F1 score from the evaluation report.
- **Quality Gate:** An automated check that prevents low-quality models from progressing in the pipeline. The threshold (0.85 in this case) is a team decision based on business requirements.

---

## AIML Connection

The *Evaluation Metrics* reading emphasized never relying on a single metric. The ConditionStep here checks F1 score, but in production you would add multiple conditions -- F1 AND precision AND recall AND no regression from the baseline. The `conditions` parameter of ConditionStep accepts a list, and all conditions must be true for the `if_steps` branch to execute.
