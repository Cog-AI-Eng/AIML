# AWSSageMaker-MLOps Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** MLOps & CI/CD Principles (SM-CT-MLOps&CI/CDPrinciples), Pipelines, DAGs & Versioning (SM-CT-Pipelines,DAGs&Versioning)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | MLOps Principles and Building a SageMaker Pipeline | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Executing, Monitoring, and Adding Conditional Logic | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Model Monitor, EventBridge, and the MLOps Feedback Loop | 45 |
| Buffer | Open Q&A, Git Branch Activity, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (continued)**

The FraudShield data science team has completed Modules 1-3: they set up the SageMaker environment, trained a fraud detection model with Script Mode, registered it in the Model Registry, deployed it to a real-time endpoint, and invoked it for predictions. Every step worked -- but every step was manual.

Now the CTO asks: "We need to retrain this model every week on fresh transaction data. Can you automate the whole pipeline so we don't need a human clicking through the console every Monday morning?"

Today the team automates the FraudShield ML workflow:

1. **"How do we replace manual steps with an automated pipeline?"** (SageMaker Pipelines, DAGs)
2. **"How do we ensure only good models reach production?"** (Conditional logic, quality gates)
3. **"How do we know when the model needs retraining?"** (Model Monitor, EventBridge, feedback loop)

This scenario threads through every Module 4 exit criterion: Associates will explain MLOps and CI/CD principles, build a SageMaker Pipeline DAG, add conditional logic, explore Model Monitor conceptually, and see how EventBridge closes the automation loop.

**Code-heavy lecture.** Modules 1 and 3 were console-first. Module 2 introduced SDK code. Module 4 is primarily code because pipelines are defined programmatically. Console is used for viewing DAGs, monitoring executions, and exploring Model Monitor.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] SageMaker Studio open with a notebook ready
- [ ] SageMaker Python SDK installed (`pip install sagemaker`)
- [ ] S3 bucket with FraudShield training data from Module 2 (`data/train/train.csv`, `data/validation/validation.csv`)
- [ ] Script Mode `train.py` from Module 2 available in Studio
- [ ] Model Package Group `fraud-detection-rf` exists in the Model Registry (from Module 3)
- [ ] IAM execution role with SageMaker, S3, and CloudWatch permissions
- [ ] Screen sharing enabled, font increased for projector readability
- [ ] This instructor guide open in a second tab

### Additional Script: Preprocessing

Create `preprocess.py` before the lecture (or build it live in Stage 1):

```python
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"

    df = pd.read_csv(os.path.join(input_dir, "train.csv"))

    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["target"]
    )

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train", "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation", "validation.csv"), index=False)

    print(f"Train: {len(train_df)} rows, Validation: {len(val_df)} rows")


if __name__ == "__main__":
    main()
```

### Additional Script: Evaluation

Create `evaluate.py` before the lecture:

```python
import argparse
import json
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"

    import tarfile
    with tarfile.open(os.path.join(model_dir, "model.tar.gz"), "r:gz") as tar:
        tar.extractall(model_dir)

    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    df = pd.read_csv(os.path.join(test_dir, "validation.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]

    predictions = model.predict(X)

    metrics = {
        "metrics": {
            "accuracy": round(accuracy_score(y, predictions), 4),
            "f1": round(f1_score(y, predictions, zero_division=0), 4),
            "precision": round(precision_score(y, predictions, zero_division=0), 4),
            "recall": round(recall_score(y, predictions, zero_division=0), 4),
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Evaluation: {metrics}")


if __name__ == "__main__":
    main()
```

### Student Prerequisites

- [ ] Completed Modules 1-3 lectures (environment, training, deployment)
- [ ] Completed readings: MLOps & CI/CD Principles, Pipelines, DAGs & Versioning
- [ ] Studio notebook open with SageMaker SDK available
- [ ] `train.py` from Module 2 available

---

## Stage 1: MLOps Principles and Building a SageMaker Pipeline

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Explain the core principles of MLOps and CI/CD for machine learning (Required)
- Build a simple SageMaker Pipeline DAG for a training and deployment workflow (Preferred)

### Instructor Opening (5 minutes -- talk, no code)

> "Over the last three modules, you built the entire SageMaker ML lifecycle by hand: set up Studio, created IAM roles, wrote a Script Mode training script, configured Estimators, launched training jobs, registered models in the Registry, approved them, deployed endpoints, and invoked them. Every step worked. But every step was manual."

> "Imagine doing that every Monday morning when FraudShield's fresh transaction data arrives. You would spend hours clicking through consoles and copying S3 paths. If you forget a step -- say, you skip the evaluation and deploy a bad model -- the fraud detection system degrades and real money is at stake."

> "MLOps solves this by automating the whole pipeline. Today we will encode every manual step into a SageMaker Pipeline that runs end to end without human intervention, except for one critical gate: model approval."

---

### STEP 1 -- MLOps Principles: The Whiteboard Discussion (10 minutes)

**Pacing: interactive whiteboard discussion. No code.**

Draw three pillars on screen:

```
MLOps = Automation + Versioning & Governance + Monitoring & Feedback
```

> "These three pillars extend the reproducibility principles from AIML Foundations. In AIML, you learned consistent environments, version control, and fixed random seeds. MLOps adds automation -- the fourth pillar that makes reproducibility happen without a human in the loop."

**CI/CD adapted for ML:**

Draw the mapping table:

| Software CI/CD | ML CI/CD (MLOps) | SageMaker Tool |
|---------------|-----------------|----------------|
| Code commit triggers build | Code or data change triggers training | Pipelines, EventBridge |
| Unit tests validate code | Model evaluation validates quality | Evaluation step, Registry metrics |
| Build artifact (binary) | Model artifact (model.tar.gz) | S3, Model Registry |
| Deploy to staging/production | Deploy to endpoint | Endpoint from Registry |
| Application monitoring | Model monitoring (drift) | Model Monitor |
| Bug fix triggers new build | Drift alert triggers retraining | EventBridge + Pipeline |

> "The key difference from software CI/CD: ML has two sources of change. Code changes AND data changes can both degrade model quality. Your pipeline must handle both."

**MLOps maturity levels:**

| Level | What It Looks Like | Where We've Been |
|-------|-------------------|-----------------|
| Level 0 -- Manual | Every step by hand | Modules 1-3 |
| Level 1 -- Pipeline automation | Training/eval/registration automated; human approves | Today's lecture |
| Level 2 -- Full CI/CD | Code push triggers pipeline; auto-approval; monitoring triggers retraining | Stretch goal |

> "Today we build Level 1. The pipeline automates everything except the approval decision, which stays with a human reviewer."

[PAUSE FOR Q&A - Ask: "Why does ML need monitoring after deployment, but most software doesn't?" (Software behavior doesn't change unless code changes. Model behavior degrades when the input data distribution shifts -- data drift and concept drift.)]

---

### STEP 2 -- Setting Up Pipeline Dependencies (5 minutes)

**Pacing: live coding in a notebook cell.**

```python
# STEP 2: Pipeline setup
import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
prefix = "fraudshield"

print(f"Role: {role}")
print(f"Bucket: {bucket}")
```

> "These imports are the building blocks for every SageMaker Pipeline. `ProcessingStep` wraps preprocessing, `TrainingStep` wraps training, `RegisterModel` wraps model registration. `ParameterString` makes the pipeline reusable with different inputs."

---

### STEP 3 -- Defining Pipeline Parameters (5 minutes)

**Pacing: line-by-line.**

```python
# STEP 3: Pipeline parameters (make the pipeline reusable)
input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{bucket}/{prefix}/data/train/train.csv",
)

n_estimators = ParameterInteger(
    name="NEstimators",
    default_value=100,
)

approval_status = ParameterString(
    name="ApprovalStatus",
    default_value="PendingManualApproval",
)
```

> "Parameters make the pipeline flexible. The same pipeline definition can process different datasets, try different hyperparameters, or start with different approval statuses -- all without editing code. When you execute the pipeline, you can override any default."

> "Compare this to hardcoding `s3://my-bucket/data/train.csv` directly in the step. If you hardcode, you need a new pipeline for every dataset. With parameters, one pipeline handles them all."

---

### STEP 4 -- Building the Preprocessing Step (10 minutes)

**Pacing: line-by-line with explanations.**

> "First, let's set up the preprocessing step. This replaces the manual data preparation we did before Module 2."

```python
# STEP 4: Preprocessing step
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="fraudshield-preprocess",
)

preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/{prefix}/pipeline/processed/train/",
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination=f"s3://{bucket}/{prefix}/pipeline/processed/validation/",
        ),
    ],
    code="preprocess.py",
)
```

> "The `ProcessingStep` wraps a `SKLearnProcessor` the same way a `TrainingStep` wraps an `Estimator`. The `inputs` pull raw data from S3 into the container. The `outputs` push processed data back to S3. The `code` parameter points to our preprocessing script."

> "Notice that we use the `input_data` parameter here, not a hardcoded S3 path. When the pipeline runs, SageMaker substitutes the actual value."

---

### STEP 5 -- Building the Training Step (10 minutes)

**Pacing: line-by-line.**

```python
# STEP 5: Training step
estimator = SKLearn(
    entry_point="train.py",
    source_dir="code/",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    hyperparameters={
        "n-estimators": n_estimators,
        "random-state": 42,
    },
    output_path=f"s3://{bucket}/{prefix}/pipeline/output/",
    base_job_name="fraudshield-train",
)

train_step = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig
                .Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)
```

> "This is the same `SKLearn` Estimator from Module 2, wrapped in a `TrainingStep`. The critical line is the `inputs`: instead of a hardcoded S3 path, we reference `preprocess_step.properties`. This creates a dependency -- SageMaker knows that Train cannot start until Preprocess finishes, because Train needs Preprocess's output."

> "This dependency chain is how the DAG is built automatically. You do not draw the graph. SageMaker infers it from the data references between steps."

[PAUSE FOR Q&A - Ask: "What happens if you accidentally hardcode the S3 path in the training step input instead of referencing the preprocess step?" (The pipeline would not know about the dependency. Both steps could run in parallel, and the training step might read stale or missing data.)]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: Executing, Monitoring, and Adding Conditional Logic

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Build a simple SageMaker Pipeline DAG for a training and deployment workflow (Preferred)
- Orchestrate complex ML workflows using SageMaker Pipelines with conditional logic steps (Stretch)

### STEP 6 -- Building the Registration Step (5 minutes)

**Pacing: line-by-line.**

```python
# STEP 6: Registration step
register_step = RegisterModel(
    name="Register",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    approval_status=approval_status,
    model_package_group_name="fraud-detection-rf",
)
```

> "The `RegisterModel` step does exactly what you did manually in Module 3: it registers a new version in the `fraud-detection-rf` group. The `model_data` references the training step's output artifact. The `approval_status` uses our pipeline parameter, defaulting to `PendingManualApproval`."

> "After this step runs, a new version appears in the Model Registry -- no human clicks required. The human's only job is the approval decision."

---

### STEP 7 -- Assembling and Submitting the Pipeline (5 minutes)

**Pacing: live execution.**

```python
# STEP 7: Assemble the pipeline
pipeline = Pipeline(
    name="fraudshield-detection-pipeline",
    parameters=[input_data, n_estimators, approval_status],
    steps=[preprocess_step, train_step, register_step],
)

pipeline.upsert(role_arn=role)
print(f"Pipeline created: {pipeline.name}")
```

> "`pipeline.upsert()` creates or updates the pipeline definition in SageMaker. It does NOT execute it. Think of this as saving the recipe -- the kitchen has not started cooking yet. The pipeline is now visible in the console."

---

### STEP 8 -- Viewing the DAG in the Console (10 minutes)

**Pacing: live demonstration in the console.**

1. Navigate to **SageMaker > Pipelines > Pipelines**. Show `fraudshield-detection-pipeline` in the list.
2. Click the pipeline name. Walk through each tab:

**Graph tab:**

> "This is the visual DAG. Each box is a step. The arrows show dependencies. Preprocess feeds into Train, which feeds into Register. SageMaker inferred this graph from the data references in our code."

Point out each node and its connections.

**Parameters tab:**

> "Here are our three parameters with their defaults: InputData, NEstimators, ApprovalStatus. When you create an execution, you can override any of these."

**Executions tab:**

> "Empty for now -- we have not run it yet. Let's change that."

---

### STEP 9 -- Executing the Pipeline (10 minutes)

**Pacing: live execution. All Associates run simultaneously.**

**Execute from the console:**

1. On the pipeline details page, click **Create execution** (or **Start**).
2. Review the parameters. Keep the defaults.
3. Click **Start**.

**Or execute from the notebook:**

```python
# STEP 9: Execute the pipeline
execution = pipeline.start()
print(f"Execution: {execution.arn}")
```

> "The pipeline is now running. SageMaker executes the steps in order: Preprocess first, then Train (after Preprocess finishes), then Register (after Train finishes). Each step provisions its own compute, runs, and tears down."

4. Switch to the console. Click into the running execution.
5. Watch the DAG update in real time:
   - **Green** = Succeeded
   - **Blue** = Executing
   - **Red** = Failed

> "Watch Preprocess turn green first, then Train starts. This takes 5-10 minutes total. While we wait, let's add conditional logic to the pipeline."

**Teaching Note:** The execution runs in the background. Let Associates monitor it on their own screens while you proceed with the conditional logic discussion. Circle back to verify completion after Step 11.

---

### STEP 10 -- Adding an Evaluation Step (10 minutes)

**Pacing: line-by-line in a new notebook cell.**

> "Right now our pipeline registers every model, regardless of quality. That is like approving every pull request without code review. Let's add an evaluation step that checks model quality before registration."

```python
# STEP 10: Evaluation step
from sagemaker.workflow.properties import PropertyFile

eval_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

eval_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="fraudshield-evaluate",
)

eval_step = ProcessingStep(
    name="Evaluate",
    processor=eval_processor,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig
                .Outputs["validation"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/{prefix}/pipeline/evaluation/",
        ),
    ],
    code="evaluate.py",
    property_files=[eval_report],
)
```

> "The evaluation step takes two inputs: the model artifact from the training step and the validation data from the preprocessing step. It runs `evaluate.py`, which extracts the model, runs predictions on the validation set, and writes metrics (accuracy, F1, precision, recall) to `evaluation.json`."

> "The `PropertyFile` is key: it tells the pipeline that `evaluation.json` contains structured data that subsequent steps can read. This is what enables conditional logic."

---

### STEP 11 -- Adding a Condition Step (10 minutes)

**Pacing: line-by-line.**

```python
# STEP 11: Condition step (only register if F1 >= 0.85)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

quality_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name="Evaluate",
        property_file=eval_report,
        json_path="metrics.f1",
    ),
    right=0.85,
)

condition_step = ConditionStep(
    name="CheckQuality",
    conditions=[quality_condition],
    if_steps=[register_step],
    else_steps=[],
)
```

> "This step reads the F1 score from the evaluation report and compares it to 0.85. If F1 is at least 0.85, the pipeline proceeds to Register. If not, the pipeline ends without registering a low-quality model."

> "Remember the approval criteria table from Module 3? F1 >= 0.85 was our primary threshold. The condition step automates that check. Instead of a human looking at metrics and clicking Approve, the pipeline makes the first-pass decision automatically."

> "The `else_steps` is empty, meaning the pipeline simply stops if the model is not good enough. You could add a notification step instead -- triggering an SNS alert that says 'Model quality below threshold, investigate.'"

**Update the pipeline with the new steps:**

```python
# STEP 11b: Update the pipeline with evaluation and condition
pipeline_v2 = Pipeline(
    name="fraudshield-detection-pipeline",
    parameters=[input_data, n_estimators, approval_status],
    steps=[preprocess_step, train_step, eval_step, condition_step],
)

pipeline_v2.upsert(role_arn=role)
print("Pipeline updated with evaluation and condition steps")
```

> "Notice the steps list changed. Register is no longer a top-level step -- it is inside the condition step's `if_steps`. The DAG now has four nodes: Preprocess, Train, Evaluate, CheckQuality. Register only happens if the condition passes."

7. Switch to the console. Refresh the pipeline page. Show the updated DAG with four steps and the conditional branch.

[PAUSE FOR Q&A - Ask: "If the model's F1 is 0.84, what happens?" (The condition fails. The pipeline ends without registering. The model artifact still exists in S3, but it is not in the Registry.)]

---

### Verify First Execution (3 minutes)

> "Let's check on our first pipeline execution from Step 9."

Switch to the Executions tab. Click the execution. Verify all steps show **Succeeded** (green). Click into each step to show its details, logs link, and output paths.

> "Preprocess created the train/validation split. Train ran our Script Mode script. Register added a new version to the `fraud-detection-rf` group. If you open the Model Registry now, you will see the new version with `PendingManualApproval` status."

Navigate to **Governance > Model registry > fraud-detection-rf**. Show the new version.

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Model Monitor, EventBridge, and the MLOps Feedback Loop

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Explain the core principles of MLOps and CI/CD for machine learning (Required)
- Describe the use of Model Monitor for detecting data and model drift (Preferred)

### Instructor Opening (3 minutes)

> "We have automated training, evaluation, and registration. But who triggers the pipeline? Right now, we run it manually. In a real system, two things should trigger retraining: new data arriving and model quality degradation. That is the monitoring and feedback pillar of MLOps."

---

### STEP 12 -- Model Monitor: Conceptual Walkthrough (15 minutes)

**Pacing: interactive discussion with console exploration.**

> "Model Monitor answers the question you may have had since AIML Foundations: 'How do I know if my model's accuracy is degrading after deployment?' The answer is continuous monitoring."

Draw the four-step process:

```
1. Baseline  -->  2. Data Capture  -->  3. Monitoring Schedule  -->  4. Alerts
```

**1. Baseline:**

> "You provide a baseline dataset -- typically your training data. Model Monitor computes statistical profiles: feature means, standard deviations, distributions, data types. This baseline is the 'expected' behavior."

**2. Data Capture:**

> "When you create an endpoint, you can enable data capture in the Endpoint Configuration. SageMaker logs a percentage of incoming requests and responses to S3. This is the 'actual' behavior."

Navigate to **Inference > Endpoint configurations** in the console. Show where the data capture option appears (even if no endpoint is running).

**3. Monitoring Schedule:**

> "You create a monitoring job that runs on a schedule -- hourly or daily. The job compares captured data against the baseline. If feature distributions have shifted, if new values appear that were not in the training data, or if a column's data type has changed, Model Monitor generates a violations report."

Navigate to **Inference > Model monitoring** in the console. Show the monitoring interface (it may be empty).

**4. Alerts:**

> "When violations are detected, Model Monitor emits an event to EventBridge. That event can trigger a notification, a Lambda function, or -- critically -- your SageMaker Pipeline for retraining."

**Types of drift:**

| Drift Type | What Changed | Example |
|-----------|-------------|---------|
| **Data drift** | Input feature distributions shifted | Average transaction amount increased 30% |
| **Concept drift** | Relationship between features and target changed | Fraud patterns shifted to different transaction types |
| **Prediction drift** | Model output distribution changed | Model predicting fraud at 2x the baseline rate |

> "Data drift is the most common. If FraudShield's customer base grows to include international transactions (higher average amounts, different time zones), the model's training data no longer represents the current population. Model Monitor catches this before accuracy degrades."

**Connect to AIML Foundations:**

> "In the AIML Evaluation module, you evaluated models on holdout sets at a fixed point in time. Model Monitor extends that evaluation to a continuous stream of production data. The metrics are the same -- accuracy, F1, precision -- but now they are tracked over time."

[PAUSE FOR Q&A - Ask: "If Model Monitor detects data drift but the model's prediction accuracy has not changed, should you retrain?" (Not necessarily. Data drift does not always cause accuracy degradation. Monitor both data drift and model quality metrics. Retrain only when quality actually declines, or when drift is severe enough that degradation is likely.)]

---

### STEP 13 -- EventBridge: Connecting the Feedback Loop (10 minutes)

**Pacing: console demonstration.**

> "EventBridge is the glue that connects monitoring to automation. Let's see how a Model Monitor alert could trigger our pipeline."

1. Search "EventBridge" in the console. Open the EventBridge service.
2. Click **Rules > Create rule**.
3. Walk through the rule creation (demonstration only, do not create the rule):

   - **Rule name:** `fraudshield-drift-retrain`
   - **Event pattern:** Select **SageMaker** as the service. Filter for **Model Monitor** events (or show the JSON pattern):

```json
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Monitor Alert"]
}
```

   - **Target:** Select **SageMaker Pipeline** and choose `fraudshield-detection-pipeline`.

> "When Model Monitor detects drift and emits an event, EventBridge matches the pattern and triggers our pipeline. The pipeline trains a new model on the latest data, evaluates it, and registers it if quality passes. A human reviews the new version in the Registry and approves it. The feedback loop is complete."

4. Show the conceptual flow:

```
Endpoint serves traffic
    |
    v
Data Capture logs requests to S3
    |
    v
Model Monitor compares to baseline (scheduled)
    |
    v (drift detected)
EventBridge triggers Pipeline
    |
    v
Pipeline: Preprocess --> Train --> Evaluate --> CheckQuality --> Register
    |
    v
Human approves in Model Registry
    |
    v
Deploy new model to Endpoint
```

> "This is Level 1 MLOps. The pipeline runs automatically. The human only approves or rejects in the Registry. In Level 2, even the approval step would be automated for routine cases."

---

### STEP 14 -- EventBridge Notification Setup (5 minutes)

**Pacing: console demonstration.**

> "Even before full automation, a simple notification rule keeps your team informed."

1. In EventBridge, show the rule creation for notifications:
   - **Event pattern:** SageMaker Model Package State Change events.
   - **Target:** SNS topic that emails the review team.

> "When a pipeline registers a new model version with `PendingManualApproval`, the team gets an email. No more versions sitting pending for days because nobody noticed. This is the 'notified' maturity level from the reading."

| Maturity | Workflow | Automation Level |
|----------|---------|-----------------|
| Manual | Human clicks Approve in console | None |
| Notified | EventBridge emails team on new pending version | Notification only |
| Automated | Lambda checks metrics and auto-approves | Full (with human override) |

---

### STEP 15 -- CI/CD with CodePipeline: Conceptual Overview (7 minutes)

**Pacing: whiteboard discussion. No code or console.**

> "The final piece is connecting code changes to the pipeline. Right now, if you update `train.py` -- say, switching from Random Forest to Gradient Boosting -- you have to manually run `pipeline.upsert()` and `pipeline.start()`. AWS CodePipeline automates that."

Draw the flow:

```
Developer pushes code to Git
    |
    v
CodePipeline detects change
    |
    v
CodeBuild runs tests, calls pipeline.upsert() and pipeline.start()
    |
    v
SageMaker Pipeline executes (Preprocess --> Train --> Evaluate --> Register)
    |
    v
Model Registry (PendingManualApproval)
    |
    v
Human approves
    |
    v
Deployment step deploys to endpoint
```

> "SageMaker Pipelines orchestrates ML steps. CodePipeline orchestrates software delivery steps. They work together: CodePipeline triggers SageMaker Pipeline when code changes, and SageMaker Pipeline triggers deployment when a model is approved."

> "This is a stretch goal for this curriculum. The concept is more important than the implementation right now: every tool you have learned -- Git, Pipelines, Registry, EventBridge -- connects into a single automated system."

Navigate to the console. Search "CodePipeline." Show the service page (even if no pipelines exist). Point out that it is a separate service from SageMaker Pipelines.

---

### STEP 16 -- Full Lifecycle Consolidation (5 minutes)

**Pacing: interactive discussion.**

> "Let's map every tool from the entire curriculum to the MLOps lifecycle."

| Lifecycle Stage | Manual (Modules 1-3) | Automated (Module 4) |
|----------------|---------------------|---------------------|
| **Prepare** | Upload data to S3 manually | ProcessingStep in Pipeline |
| **Build** | Write training script in Studio | Git + CodeBuild |
| **Train & Tune** | `estimator.fit()` in notebook | TrainingStep in Pipeline |
| **Evaluate** | Print metrics in notebook | Evaluation ProcessingStep + ConditionStep |
| **Register** | `estimator.register()` in notebook | RegisterModel step in Pipeline |
| **Approve** | Click Approve in console | Human approval (Level 1) or Lambda (Level 2) |
| **Deploy** | Three-object pattern in console | Deployment step or manual from Registry |
| **Monitor** | (Not covered in Modules 1-3) | Model Monitor + EventBridge |
| **Retrain** | (Manual -- start over) | EventBridge triggers Pipeline |

> "The left column is where you started. The right column is where you are now. Every manual step has an automated equivalent."

[PAUSE FOR Q&A]

---

## Wrap-up & Git Branch Activity

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you built a SageMaker Pipeline that automates preprocessing, training, evaluation, and conditional registration. You explored Model Monitor conceptually and saw how EventBridge closes the feedback loop. You now understand the three pillars of MLOps -- automation, versioning, and monitoring -- and can map every SageMaker tool to its role in the lifecycle."

> "This is the capstone of the curriculum. Over four modules, you progressed from setting up a SageMaker environment to building an automated ML pipeline with quality gates. Every concept from AIML Foundations -- the ML lifecycle, reproducibility, evaluation metrics, algorithm selection -- connected to a SageMaker tool or practice. The skills you have built are directly applicable to production ML engineering."

### Git Branch Activity (20 minutes)

> "For the remainder of this session, extend the pipeline you built today."

**Activity Instructions:**

1. In Studio:

```bash
cd ~/fraudshield-ml
git checkout -b feature/pipeline-enhancement
```

2. Choose one of the following enhancements:

**Option A: Add a notification step.**
- After the ConditionStep, add a step that writes the evaluation metrics to a known S3 location (e.g., `s3://<bucket>/fraudshield/pipeline/reports/latest.json`).
- This simulates what a notification target would read.

**Option B: Add a second condition.**
- Add a precision threshold (>= 0.80) to the ConditionStep in addition to the F1 threshold.
- Both conditions must pass for registration to proceed.

**Option C: Parameterize the threshold.**
- Add a `ParameterFloat` for the F1 threshold so it can be overridden at execution time.
- Default to 0.85 but allow the team to lower it for experimental deployments.

3. Update the pipeline with `pipeline.upsert()` and verify the new DAG in the console.

4. Commit:

```bash
git add .
git commit -m "Enhance pipeline with [chosen option]"
```

> "Each option practices a different skill: Option A builds towards notification automation, Option B practices multi-condition quality gates, Option C practices pipeline parameterization. All three are patterns you will encounter in production MLOps."

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| `pipeline.upsert()` fails with permission error | The execution role needs `sagemaker:CreatePipeline` and `sagemaker:UpdatePipeline` permissions. Add via IAM console. |
| Pipeline execution stuck on Preprocess step | Check the `preprocess.py` script path. The `code` parameter must point to a valid file in the same directory or a specified `source_dir`. |
| DAG not showing in the console after upsert | Refresh the Pipelines page. Sometimes the console cache takes a few seconds to update. |
| ConditionStep always fails | Check the `json_path` in `JsonGet`. It must match the structure of `evaluation.json` exactly (e.g., `"metrics.f1"` not `"f1"`). |
| PropertyFile not found | Ensure the evaluation step's `ProcessingOutput` `output_name` matches the `PropertyFile`'s `output_name`. |
| Associates confused by Pipelines vs. CodePipeline | SageMaker Pipelines orchestrates ML steps. CodePipeline orchestrates software delivery steps. They are different services that work together. |
| Execution takes too long | Each step provisions its own instance. A 3-step pipeline can take 15-20 minutes total. This is normal. Monitor in the console. |
