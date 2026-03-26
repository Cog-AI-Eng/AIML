# Guide 2: Build and Run a SageMaker Pipeline

A SageMaker Pipeline is a DAG (Directed Acyclic Graph) of steps that automates your ML workflow. Each step runs on ephemeral compute -- instances are provisioned for each step and terminated when the step completes. In this guide, you will define a three-step pipeline in a Studio notebook, register it with SageMaker, and start an execution.

---

## Steps

### Step 1 -- Open a Studio Notebook

1. From the **SageMaker console**, open **Studio** via your domain.
2. Create a new notebook or reuse an existing one.
3. Select a **Python 3 (Data Science)** kernel.

### Step 2 -- Define Pipeline Parameters

In the first cell, define the configurable parameters for your pipeline:

```python
import sagemaker
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = "fraudshield-training-data-<your-initials>"  # UPDATE THIS

# Pipeline parameters -- can be changed per execution without editing code
instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
n_estimators = ParameterInteger(name="NEstimators", default_value=100)
```

Run this cell. Note: `ParameterString` and `ParameterInteger` create parameterized values that can be overridden when starting each pipeline execution.

### Step 3 -- Define the Processing Step

The processing step preprocesses raw data into train/test splits:

```python
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

# Create a processor
processor = SKLearnProcessor(
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
)

# Define the processing step
process_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/data/train/",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/pipeline/processed/train/",
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/pipeline/processed/test/",
        ),
    ],
    code="preprocess.py",  # You will create this script
)
```

### Step 4 -- Create the Preprocessing Script

In a new cell, create the preprocessing script:

```python
%%writefile preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

input_dir = "/opt/ml/processing/input"
train_output_dir = "/opt/ml/processing/output/train"
test_output_dir = "/opt/ml/processing/output/test"

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
df = pd.concat([pd.read_csv(os.path.join(input_dir, f)) for f in csv_files])

# Drop transaction_id (not a model feature)
df = df.drop(columns=["transaction_id"], errors="ignore")

# Stratified split to preserve the ~5% fraud ratio in both sets
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["is_fraud"]
)

train_df.to_csv(os.path.join(train_output_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(test_output_dir, "test.csv"), index=False)

print(f"Train: {len(train_df)} samples ({train_df['is_fraud'].mean():.1%} fraud)")
print(f"Test:  {len(test_df)} samples ({test_df['is_fraud'].mean():.1%} fraud)")
```

### Step 5 -- Define the Training Step

```python
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

# Create an estimator
estimator = SKLearn(
    entry_point="train.py",  # Reuse the train.py from Module 2
    role=role,
    instance_type=instance_type,  # Uses the pipeline parameter
    instance_count=1,
    framework_version="1.2-1",
    hyperparameters={"n-estimators": n_estimators, "random-state": 42},
    output_path=f"s3://{bucket}/pipeline/output",
    base_job_name="fraudshield-pipeline",
)

# Define the training step
train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=process_step.properties.ProcessingOutputConfig
                .Outputs["train"].S3Output.S3Uri,
        ),
    },
)
```

Note how `train_step` uses `process_step.properties` to get the output location of the processing step. This creates the dependency: TrainModel runs AFTER PreprocessData.

### Step 6 -- Define the Register Step

```python
from sagemaker.workflow.step_collections import RegisterModel

register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    model_package_group_name="fraud-detection-rf",
    approval_status="PendingManualApproval",
)
```

### Step 7 -- Assemble and Run the Pipeline

```python
from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name="fraudshield-detection-pipeline",
    parameters=[instance_type, n_estimators],
    steps=[process_step, train_step, register_step],
)

# Register the pipeline definition with SageMaker
pipeline.upsert(role_arn=role)
print("Pipeline upserted successfully")

# Start an execution
execution = pipeline.start()
print(f"Pipeline execution started: {execution.describe()['PipelineExecutionArn']}")
```

Run this cell. The pipeline definition will be registered, and an execution will start.

### Step 8 -- Switch to the Console

Now switch to the SageMaker console to monitor the execution in Guide 3. Keep the notebook tab open -- you will return to it in Guide 4.

---

## Presentation Checkpoint

Be prepared to show:
- The notebook cells with the pipeline definition
- Explain: What is a Pipeline Parameter and why is it useful? (It allows you to change values like instance type or hyperparameters per execution without editing the pipeline code. This makes the pipeline reusable across different experiments.)
- Explain: How does the TrainingStep know where to find the preprocessed data? (It uses `process_step.properties` to dynamically reference the output location of the processing step. SageMaker resolves this at runtime.)
- Explain: What is the difference between `pipeline.upsert()` and `pipeline.start()`? (Upsert registers or updates the pipeline definition. Start creates and runs a new execution of that definition.)

---

## Key Concepts

- **DAG (Directed Acyclic Graph):** A graph of steps where each step depends on zero or more previous steps. "Directed" means dependencies flow in one direction; "acyclic" means there are no circular dependencies.
- **Step Properties:** Each step exposes properties (like output locations) that downstream steps can reference. This is how SageMaker builds the dependency graph automatically.
- **`pipeline.upsert()`:** Creates a new pipeline definition or updates an existing one. Like "git push" for your pipeline code.
- **Ephemeral Compute:** Each pipeline step runs on its own instance that is provisioned at the start of the step and terminated at the end. No persistent compute runs between steps.
