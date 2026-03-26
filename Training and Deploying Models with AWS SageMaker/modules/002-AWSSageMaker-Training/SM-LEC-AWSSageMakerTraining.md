# AWSSageMaker-Training Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** BYOM & BYOS Approaches (SM-CT-BYOM&BYOSApproaches), Script Mode Structure (SM-CT-ScriptModeStructure), Estimators & Configurations (SM-CT-Estimators&Configurations), Training Job Anatomy (SM-CT-TrainingJobAnatomy), Model Artifacts & S3 Storage (SM-CT-ModelArtifacts&S3Storage)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | From Local Script to Script Mode: Adapting AIML Code for SageMaker | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Configuring and Launching a Training Job: Console and SDK | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Anatomy of a Live Training Job: Debugging, Artifacts, and Local Mode | 45 |
| Buffer | Open Q&A, Git Branch Activity, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (continued)**

Associates continue as ML engineers at FraudShield. In Module 1, the team set up their SageMaker environment and deployed a JumpStart model to prove the platform works. Now leadership wants a custom fraud detection model trained on FraudShield's proprietary transaction data. The data science team has already built a working scikit-learn Random Forest classifier locally (mirroring what Associates did in AIML Foundations). Today's mission: move that local training script onto SageMaker's managed infrastructure so it runs at scale, stores artifacts automatically, and is fully reproducible.

1. **"How do we adapt our local training script for SageMaker?"** (Script Mode structure)
2. **"How do we configure and launch the training job?"** (Estimators, console, SDK)
3. **"How do we know what happened during training, and where is the model?"** (Training job anatomy, artifacts, S3)

This scenario threads through every Module 2 exit criterion: Associates will differentiate BYOM from Script Mode, write a Script Mode training script with the required structure, configure an Estimator, launch a training job, inspect its lifecycle through the console, and locate the model artifact in S3.

**Why code in this lecture?** Module 1 was console-only. Module 2 introduces the SDK because training jobs require script files and Estimator configuration. However, every SDK operation is preceded by its console equivalent so Associates see both sides.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] SageMaker Studio Domain running (created in Module 1 lecture)
- [ ] S3 bucket created with training data uploaded:
  - Bucket: `fraudshield-training-data` (or similar)
  - Prefix: `data/train/` containing `train.csv`
  - Prefix: `data/validation/` containing `validation.csv`
- [ ] Training data is a simple tabular CSV with columns including `target` (binary: 0/1 for fraud/non-fraud), 5-8 numeric feature columns
- [ ] IAM execution role `SageMaker-Training-LeastPrivilege` exists (created in Module 1 lecture) with S3 access to the training bucket
- [ ] Studio notebook open with SageMaker Python SDK installed (`pip install sagemaker`)
- [ ] Local training script from AIML Foundations available for reference
- [ ] Screen sharing enabled, font size increased for projector readability
- [ ] This instructor guide open in a second tab

### Recommended Data Preparation

Generate a synthetic fraud dataset in a notebook before the lecture (or do it live in the first 5 minutes):

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "amount": np.random.exponential(500, n).round(2),
    "hour": np.random.randint(0, 24, n),
    "distance_from_home": np.random.exponential(50, n).round(2),
    "transaction_count_24h": np.random.poisson(5, n),
    "is_international": np.random.binomial(1, 0.1, n),
    "merchant_risk_score": np.random.uniform(0, 1, n).round(3),
})
data["target"] = ((data["amount"] > 800) & (data["hour"] < 6) | (data["merchant_risk_score"] > 0.85)).astype(int)

train = data.iloc[:1600]
val = data.iloc[1600:]
train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
```

Upload the CSVs to S3 before class:

```bash
aws s3 cp train.csv s3://fraudshield-training-data/data/train/train.csv
aws s3 cp validation.csv s3://fraudshield-training-data/data/validation/validation.csv
```

### Student Prerequisites

- [ ] Completed Module 1 lecture (working Studio Domain and IAM role)
- [ ] Completed readings: BYOM & BYOS Approaches, Script Mode Structure, Estimators & Configurations, Training Job Anatomy, Model Artifacts & S3 Storage
- [ ] Studio notebook open with SageMaker SDK available

---

## Stage 1: From Local Script to Script Mode -- Adapting AIML Code for SageMaker

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Differentiate between Bring-Your-Own-Model (BYOM) and Bring-Your-Own-Script (Script Mode) approaches (Required)
- Architect a SageMaker training script using the required Script Mode structure (main guard, parser, output paths) (Required)

### Instructor Opening (5 minutes -- talk, no code)

> "In Module 1 you deployed a JumpStart model -- someone else's model on SageMaker's infrastructure. Today we write our own. Think back to the AIML Foundations module: you loaded a CSV, trained a Random Forest, evaluated it, and saved it with `joblib.dump()`. That script ran on your laptop. Our goal today is to take that same script and run it on SageMaker's managed infrastructure so it trains in the cloud, stores artifacts in S3, and is reproducible for the whole team."

> "FraudShield's data science team has a working local training script. Your job is to adapt it for SageMaker, configure the training infrastructure, launch the job, and verify the output. By the end of this lecture, you will have run a real training job and found your model artifact in S3."

---

### STEP 1 -- BYOM vs Script Mode: Setting the Context (5 minutes)

**Pacing: whiteboard/slide discussion.** No code yet.

Draw the customization spectrum on screen:

```
Built-in Algorithms ←——→ Script Mode (BYOS) ←——→ Bring Your Own Container (BYOM)
   (least control)         (recommended)            (most control)
```

> "SageMaker gives you three levels of customization. Built-in algorithms: you provide only data and hyperparameters. Script Mode: you provide a training script, SageMaker provides the container. BYOM: you provide the entire Docker image."

| Approach | You Provide | SageMaker Provides | Best For |
|----------|------------|-------------------|----------|
| Built-in algorithms | Data + hyperparameters | Everything else | Standard problems, no custom code |
| Script Mode (BYOS) | Training script (.py) | Managed container + framework | Custom training with standard frameworks |
| BYOM | Complete Docker image | Compute infrastructure only | Proprietary dependencies or frameworks |

> "For this curriculum and for most real-world projects, Script Mode is the right choice. You already know scikit-learn from AIML Foundations. SageMaker has a managed container with scikit-learn pre-installed. You just supply your script."

[PAUSE FOR Q&A - Ask: "When would you choose BYOM over Script Mode?" (Custom C++ extensions, proprietary frameworks, specific CUDA versions, compliance-mandated base images)]

---

### STEP 2 -- The Local Training Script (5 minutes)

**Pacing: line-by-line in a Studio notebook.** Open a new notebook cell.

> "Let's start with the script you already know. This is essentially what you wrote in AIML Foundations."

```python
# STEP 2: The local training script (what we're starting from)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("data/train.csv")
X = data.drop("target", axis=1)
y = data["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print(f"Training accuracy: {accuracy_score(y, model.predict(X)):.4f}")

joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
```

> "Three things are hardcoded here: the data path, the hyperparameters, and the model save location. SageMaker needs all three to be flexible so the same script can run with different data, different settings, and save to the right place. Let's fix that."

**Instructor Note:** Do NOT run this cell. It is for visual comparison only. If you have `train.csv` locally, you can run it to show the output, but the point is to contrast it with the Script Mode version.

---

### STEP 3 -- Building the Script Mode Version (20 minutes)

**Pacing: line-by-line.** Create a new file in Studio called `train.py`. Type each section individually, pausing to explain.

> "We need to make three changes. I'm going to build this file from scratch so you see exactly where each change goes."

**Part A: Imports and main guard (3 minutes)**

```python
# STEP 3A: Imports and main guard
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


if __name__ == "__main__":
    main()
```

> "Two new imports: `argparse` for hyperparameters, `os` for environment variables. The `if __name__` guard is mandatory -- SageMaker executes your script as a module, and without this guard, your code runs during import, which breaks things."

**Part B: The argument parser (5 minutes)**

Add the `main()` function above the guard:

```python
# STEP 3B: Argument parser for hyperparameters
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
```

> "Instead of hardcoding `n_estimators=100`, we accept it as a command-line argument. When SageMaker runs this script, it reads the hyperparameters from the training job configuration and passes them as `--n-estimators 100 --random-state 42` on the command line. Our parser catches them."

> "Why does this matter? Reproducibility. In AIML Foundations you learned that fixed random seeds and versioned configurations make experiments repeatable. The argument parser means you can run the same script with different hyperparameters without editing code. SageMaker records which values were used in the training job metadata."

[PAUSE FOR Q&A - Ask: "What happens if you define `--n-estimators` in the parser but pass `n_estimators` (with underscores) in the job config?" (SageMaker converts between hyphens and underscores, so both work)]

**Part C: Environment variables for data and model paths (5 minutes)**

```python
    # STEP 3C: Standard directory paths via environment variables
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
```

> "On SageMaker, `SM_CHANNEL_TRAIN` points to `/opt/ml/input/data/train` -- the directory where SageMaker downloads your S3 training data before your script runs. `SM_MODEL_DIR` points to `/opt/ml/model` -- the directory where you save your trained model. SageMaker compresses that directory into `model.tar.gz` and uploads it to S3."

> "The fallback defaults (`/opt/ml/input/data/train`) let you test this script locally too, as long as you create matching directories."

Draw the container directory layout:

```
/opt/ml/
├── input/
│   └── data/
│       ├── train/        ← SM_CHANNEL_TRAIN (your CSV lands here)
│       └── validation/   ← SM_CHANNEL_VALIDATION
├── model/                ← SM_MODEL_DIR (save your model here)
└── output/
    └── failure           ← write error message on failure
```

**Part D: Training logic and model saving (5 minutes)**

```python
    # STEP 3D: Training logic (same as local, different paths)
    data = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X = data.drop("target", axis=1)
    y = data["target"]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model.fit(X, y)

    accuracy = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {accuracy:.4f}")

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
```

> "The training logic is identical to the local version. Three things changed: data path uses `train_dir` from the environment variable, hyperparameters come from `args` instead of hardcoded values, and the model saves to `model_dir` instead of a local path. That's it."

**Part E: Final complete script (2 minutes)**

Show the complete file side-by-side with the local version:

```python
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    data = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X = data.drop("target", axis=1)
    y = data["target"]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model.fit(X, y)

    accuracy = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {accuracy:.4f}")

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
```

> "Compare this with the local script: same imports (plus `argparse` and `os`), same algorithm, same logic. Three structural changes. This is the Script Mode pattern you will use for every training script in this curriculum."

**Teaching Note:** Some Associates may ask about `SM_CHANNEL_VALIDATION`. Explain: if you configure a validation channel, SageMaker sets `SM_CHANNEL_VALIDATION`. Your script can read validation data, compute validation metrics, and print them. We keep this script simple for now; the assignment will add validation.

---

### STEP 4 -- Uploading the Script to S3 (5 minutes)

**Pacing: live demonstration in a notebook cell.**

```python
# STEP 4: Upload the training script to S3
import sagemaker
import boto3

session = sagemaker.Session()
bucket = session.default_bucket()
script_prefix = "fraudshield/code"

s3 = boto3.client("s3")
s3.upload_file("train.py", bucket, f"{script_prefix}/train.py")
print(f"Script uploaded to s3://{bucket}/{script_prefix}/train.py")
```

> "SageMaker needs your script in S3 so it can download it into the training container. When you use the SDK's `source_dir` parameter, it handles the upload automatically. But I want you to see the manual step first so you understand what is happening."

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: Configuring and Launching a Training Job -- Console and SDK

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Configure a SageMaker Estimator with appropriate instance types and hyperparameters (Required)
- Describe the anatomy of a SageMaker Training Job (Input channels, Docker containers, S3 locations) (Required)

### STEP 5 -- Console Walkthrough: Creating a Training Job (15 minutes)

**Pacing: live demonstration.** Navigate the console while explaining each field. Do NOT submit the job from the console -- this is a dry run to map fields to SDK parameters.

> "Before we write any SDK code, let's see every parameter on the console form. This builds the mental model that every SDK parameter maps to a form field."

1. Navigate to **SageMaker > Training > Training jobs**. Click **Create training job**.

2. **Job settings:**
   - **Job name:** `fraudshield-rf-v1`
   - **IAM role:** Select `SageMaker-Training-LeastPrivilege` (or the default Domain role)

> "The job name should be descriptive. Every training job gets a unique name, and this is how you will find it later in the console, CloudWatch logs, and S3."

3. **Algorithm source:**
   - Select **SageMaker built-in algorithm**.
   - Choose **Scikit Learn** and the latest version.
   - **Entry point:** `train.py`
   - **Source directory:** The S3 path where you uploaded the script.

> "This tells SageMaker: 'Use the managed scikit-learn container, download my script from S3, and run `train.py` inside it.'"

4. **Resource configuration:**
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`
   - **Volume size:** `30` GB
   - **Max runtime:** `3600` seconds

> "Instance type is where cost control lives. `ml.m5.xlarge` is a general-purpose instance within Free Tier limits. A GPU instance like `ml.p3.2xlarge` costs roughly 10x more per hour. For scikit-learn, which does not use GPUs, there is no benefit to a GPU instance."

5. **Hyperparameters:**
   - Add `n-estimators` = `100`
   - Add `random-state` = `42`

> "These key names must match what your `argparse` parser expects. SageMaker passes them as command-line arguments to your script."

6. **Input data configuration:**
   - Channel name: `train`
   - S3 location: `s3://fraudshield-training-data/data/train/`
   - Input mode: `File`

> "The channel name `train` maps to `SM_CHANNEL_TRAIN` in your script. SageMaker downloads everything from this S3 prefix into `/opt/ml/input/data/train/` before your script runs."

7. **Output data configuration:**
   - S3 output path: `s3://fraudshield-training-data/output/`

> "After training, SageMaker compresses `/opt/ml/model/` into `model.tar.gz` and uploads it to this path under a subdirectory named after the training job."

8. **Do NOT click Create.** Return to the notebook.

> "Now you have seen every field. Let's do the same thing in code."

---

### STEP 6 -- SDK Estimator Configuration (10 minutes)

**Pacing: line-by-line in a notebook cell.** Map each parameter to the console field you just showed.

```python
# STEP 6: Configure the Estimator (SDK equivalent of the console form)
import sagemaker
from sagemaker.sklearn import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

estimator = SKLearn(
    entry_point="train.py",                    # Console: Entry point
    source_dir="code/",                        # Console: Source directory
    role=role,                                 # Console: IAM role
    instance_type="ml.m5.xlarge",              # Console: Instance type
    instance_count=1,                          # Console: Instance count
    volume_size=30,                            # Console: Volume size
    max_run=3600,                              # Console: Max runtime
    framework_version="1.2-1",                 # Console: Framework version
    hyperparameters={                          # Console: Hyperparameters
        "n-estimators": 100,
        "random-state": 42,
    },
    output_path=f"s3://{bucket}/fraudshield/output/",  # Console: S3 output path
    base_job_name="fraudshield-rf",            # Console: Job name prefix
)
```

> "Every line maps to a form field. The `SKLearn` class tells SageMaker to use the managed scikit-learn container. `entry_point` is your script filename. `source_dir` is the local directory containing your script -- the SDK uploads it to S3 automatically. `hyperparameters` are the key-value pairs."

Walk through the parameter-to-field mapping table:

| Console Field | SDK Parameter | Our Value |
|--------------|--------------|-----------|
| Job name | `base_job_name` | `fraudshield-rf` |
| IAM role | `role` | Execution role ARN |
| Framework | `SKLearn` class + `framework_version` | scikit-learn 1.2-1 |
| Entry point | `entry_point` | `train.py` |
| Source directory | `source_dir` | `code/` |
| Instance type | `instance_type` | `ml.m5.xlarge` |
| Instance count | `instance_count` | `1` |
| Volume size | `volume_size` | `30` GB |
| Max runtime | `max_run` | `3600` seconds |
| Hyperparameters | `hyperparameters` | `n-estimators=100, random-state=42` |
| S3 output path | `output_path` | `s3://.../fraudshield/output/` |

[PAUSE FOR Q&A - Ask: "If you wanted to train with PyTorch instead of scikit-learn, what would you change?" (Import `sagemaker.pytorch.PyTorch` instead of `SKLearn`, change the `framework_version`)]

---

### STEP 7 -- Launching the Training Job (10 minutes)

**Pacing: live execution.** Run this cell and let the job start. All Associates should run simultaneously.

```python
# STEP 7: Launch the training job
estimator.fit(
    {
        "train": f"s3://{bucket}/fraudshield/data/train/",
    },
    wait=True,
    logs="All",
)
```

> "The dictionary keys are the channel names. `train` maps to `SM_CHANNEL_TRAIN` in your script. Calling `.fit()` submits the training job to SageMaker -- this is exactly what clicking 'Create training job' does in the console."

> "`wait=True` means this cell blocks until the job finishes. `logs='All'` streams CloudWatch logs directly into this notebook. You will see real-time output from your script."

The job will take 3-7 minutes. While waiting, narrate what is happening:

> "Watch the log output. Right now SageMaker is provisioning an `ml.m5.xlarge` instance -- you can see the status change to 'Starting'. Next it will pull the scikit-learn Docker container from ECR. Then it downloads your training data from S3 into `/opt/ml/input/data/train/`. Then it runs your script. You should see your `print()` statements appear: training accuracy, model saved. After that, SageMaker compresses `/opt/ml/model/` into `model.tar.gz`, uploads it to S3, and tears down the instance."

**Teaching Note:** If the job fails, do NOT panic. Common first-time failures: wrong S3 path (FileNotFoundError), missing role permissions (Access Denied), wrong script filename (entry point not found). Debug live to demonstrate the troubleshooting workflow from the Training Job Anatomy reading.

---

### STEP 8 -- Monitoring the Job in the Console (10 minutes)

**Pacing: live demonstration.** While the job runs (or immediately after completion), switch to the console.

1. Navigate to **SageMaker > Training > Training jobs**. Find `fraudshield-rf-*` in the list.
2. Click the job name. Walk through the details page:

**Job summary:**
> "Status shows Completed (or InProgress if it is still running). Training duration shows the billable time. The training image URI is the ECR path of the scikit-learn container. The IAM role ARN is the role we configured."

**Input data configuration:**
> "Here is our `train` channel pointing to the S3 prefix. This confirms the job read from the right location."

**Hyperparameters:**
> "Here are `n-estimators: 100` and `random-state: 42`. SageMaker recorded what we passed. For reproducibility, you can always come back here and see exactly what settings produced this model."

**Monitoring:**
> "CPU utilization, memory usage, disk I/O. For a small scikit-learn job like ours, CPU should be moderate and GPU should be zero. If you were training a deep learning model and GPU was at 0%, that would indicate a configuration problem."

**View logs:**
> "Click this link to open CloudWatch Logs. You see the same output that streamed into your notebook: your script's `print()` statements, accuracy metrics, and the model save confirmation."

3. Click the **View logs** link. Show the CloudWatch Logs interface.

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Anatomy of a Live Training Job -- Debugging, Artifacts, and Local Mode

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Describe the anatomy of a SageMaker Training Job (Input channels, Docker containers, S3 locations) (Required)
- Identify the location and format of model artifacts stored in S3 after training (Required)
- Configure local mode training for fast iteration and debugging (Preferred)

### STEP 9 -- Training Job Lifecycle Walkthrough (10 minutes)

**Pacing: interactive discussion with console open.** Reference the job you just ran.

> "Let's map what we just observed to the seven-step training job lifecycle."

| Step | What Happened | Evidence |
|------|--------------|----------|
| 1. Provisioning | SageMaker allocated an `ml.m5.xlarge` | Log: "Starting - Starting the training job..." |
| 2. Container pull | Pulled the scikit-learn 1.2-1 image from ECR | Training image URI on the details page |
| 3. Data download | Downloaded `train.csv` from S3 to `/opt/ml/input/data/train/` | Input data config section |
| 4. Script execution | Ran `train.py` with hyperparameters as CLI args | Log: training accuracy output |
| 5. Artifact upload | Compressed `/opt/ml/model/` to `model.tar.gz`, uploaded to S3 | Output data config section |
| 6. Logging | Streamed stdout/stderr to CloudWatch | CloudWatch Logs link |
| 7. Teardown | Terminated instance, released resources | Status: Completed |

> "Steps 1-3 are infrastructure setup. Step 4 is your code running. Steps 5-7 are cleanup. You are only billed for the time from step 1 through step 7."

---

### STEP 10 -- Finding the Model Artifact in S3 (10 minutes)

**Pacing: live demonstration.** Two paths to the artifact.

**Path A: From the training job details page.**

1. On the training job details page, scroll to **Output data configuration**.
2. Click the S3 link. The S3 console opens at the output directory.
3. Click into `output/`. Show `model.tar.gz`.

> "Every training job stores its artifact under `<output_path>/<job-name>/output/model.tar.gz`. The job name makes each artifact unique, so multiple runs never overwrite each other."

**Path B: From the SDK.**

```python
# STEP 10: Get the artifact path from the SDK
artifact_path = estimator.model_data
print(f"Model artifact: {artifact_path}")
```

> "The SDK gives you the full S3 URI. You can pass this directly to Model Registry in Module 3 or to a deployment configuration."

**Downloading and inspecting:**

```python
# STEP 10b: Download and inspect the artifact
import tarfile
import boto3

s3 = boto3.client("s3")
s3.download_file(
    bucket,
    artifact_path.replace(f"s3://{bucket}/", ""),
    "model.tar.gz",
)

with tarfile.open("model.tar.gz", "r:gz") as tar:
    tar.extractall("./extracted_model/")
    print("Contents:", tar.getnames())
```

> "Inside you will see `model.pkl` -- exactly what our script saved with `joblib.dump()`. This is the file that the Deploy stage will load into a serving container. If you ever see unexpected files here, or if `model.pkl` is missing, check your script's saving logic."

**Load and test locally:**

```python
# STEP 10c: Verify the model works
import joblib

model = joblib.load("./extracted_model/model.pkl")
sample = [[500, 3, 25, 5, 0, 0.4]]
prediction = model.predict(sample)
print(f"Prediction for sample: {prediction}")
```

> "We just round-tripped: trained on SageMaker, artifact to S3, downloaded, loaded locally, made a prediction. This verification step prevents many deployment failures in Module 3."

[PAUSE FOR Q&A - Ask: "What would the artifact contents look like if you were training a PyTorch model instead of scikit-learn?" (model.pth, possibly config.json, tokenizer files)]

---

### STEP 11 -- Debugging a Failed Training Job (10 minutes)

**Pacing: live demonstration.** Intentionally trigger a failure to show the debugging workflow.

> "Let's break something on purpose so you see the debugging workflow."

```python
# STEP 11: Intentionally trigger a failure (wrong S3 path)
bad_estimator = SKLearn(
    entry_point="train.py",
    source_dir="code/",
    role=role,
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
    hyperparameters={"n-estimators": 100, "random-state": 42},
    output_path=f"s3://{bucket}/fraudshield/output/",
    base_job_name="fraudshield-rf-debug",
)

bad_estimator.fit(
    {"train": f"s3://{bucket}/fraudshield/WRONG-PATH/"},
    wait=True,
    logs="All",
)
```

> "I deliberately pointed the `train` channel to a path that does not exist in S3. Watch what happens."

Wait for the failure. Then walk through the debugging steps:

1. **Read the error in the notebook output.** It should show a `FileNotFoundError` or empty directory error.
2. **Navigate to the failed job in the console.** Status: **Failed**.
3. **Check the error message** on the details page.
4. **Click View logs.** Show the full traceback in CloudWatch.
5. **Check Input data configuration.** The wrong S3 path is visible here.

> "The debugging workflow: read the error, check the console details page, check CloudWatch logs, verify input/output paths. The most common failures are wrong S3 paths, missing permissions, and missing Python dependencies. The Training Job Anatomy reading has a table mapping symptoms to causes."

| Symptom | Cause | Fix |
|---------|-------|-----|
| FileNotFoundError | Wrong S3 path or missing channel | Check input data config |
| Access Denied | Role lacks S3 permissions | Check IAM role policies |
| ModuleNotFoundError | Missing Python package | Add `requirements.txt` to `source_dir` |
| Script exits non-zero | Bug in training script | Read CloudWatch traceback |

**Teaching Note:** If time allows and the first job succeeded, you can demonstrate the permissions failure by temporarily removing S3 access from the role. This makes the debugging exercise more realistic.

---

### STEP 12 -- Local Mode for Fast Iteration (10 minutes)

**Pacing: live demonstration in notebook.**

> "Waiting 3-5 minutes for provisioning on every code change is slow during development. Local Mode lets you test your Script Mode script on your own machine using Docker, without submitting a job to SageMaker infrastructure."

```python
# STEP 12: Local Mode (fast iteration)
local_estimator = SKLearn(
    entry_point="train.py",
    source_dir="code/",
    role=role,
    instance_type="local",              # <-- the only change
    framework_version="1.2-1",
    hyperparameters={"n-estimators": 50, "random-state": 42},
)

local_estimator.fit({"train": "file://./data/train/"})
```

> "Two changes: `instance_type` is `'local'` instead of `'ml.m5.xlarge'`, and the data path uses `file://` instead of `s3://`. Everything else is identical. The SDK runs the same Docker container on your machine. When your script works locally, switch back to a cloud instance type and S3 paths for the real run."

> "This is the development workflow: iterate with Local Mode until your script is correct, then switch to `ml.m5.xlarge` for the production run. It saves time and money."

**Instructor Note:** Local Mode requires Docker to be installed. In Studio, Docker is available. On local machines, Associates need Docker Desktop. If Docker is not available, demonstrate the concept but skip execution. The key takeaway is the `instance_type="local"` pattern.

[PAUSE FOR Q&A - Ask: "What two parameters do you change to switch from Local Mode to cloud mode?" (instance_type from 'local' to 'ml.m5.xlarge', and data path from 'file://' to 's3://')]

---

### STEP 13 -- Lifecycle Summary and Console Verification (5 minutes)

**Pacing: interactive consolidation.**

> "Let's map everything we did today to the ML lifecycle from Module 1."

| Lifecycle Stage | What We Did Today |
|----------------|-------------------|
| **Prepare** | Generated synthetic fraud data, uploaded to S3 |
| **Build** | Adapted local script to Script Mode structure |
| **Train & Tune** | Configured Estimator, launched training job, monitored in console |
| **Deploy** | (Next module -- but we verified the artifact is ready) |
| **Monitor** | Viewed CloudWatch logs, checked instance metrics |

> "In Module 3, the artifact we produced today becomes the input to deployment. The S3 URI you printed in Step 10 is the bridge between Train and Deploy."

Navigate to **SageMaker > Training > Training jobs**. Show both jobs (the successful one and the failed debug one). Point out how the naming convention makes it easy to identify each run.

---

## Wrap-up & Git Branch Activity

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you took a local scikit-learn training script from AIML Foundations and ran it on SageMaker's managed infrastructure. You learned Script Mode's three structural requirements: main guard, argument parser, standard directory paths. You configured an Estimator in both the console and SDK. You launched a training job, watched its lifecycle, debugged a failure, found the model artifact in S3, and tested Local Mode. In Module 3, you will register this artifact in the Model Registry and deploy it to a live endpoint."

### Git Branch Activity (20 minutes)

> "For the remainder of this session, create a feature branch and extend the training script."

**Activity Instructions:**

1. In a Studio terminal:

```bash
cd ~/fraudshield-ml
git init
git add train.py
git commit -m "Base Script Mode training script"
git checkout -b feature/add-validation
```

2. Edit `train.py` to:
   - Add a `--max-depth` hyperparameter to the argument parser.
   - Read a validation channel using `SM_CHANNEL_VALIDATION`.
   - After training, load `validation.csv`, run `model.predict()` on it, and print validation accuracy.
   - Save a `metrics.json` file to `SM_MODEL_DIR` with both training and validation accuracy.

3. Test your changes by running the script locally (or with Local Mode if Docker is available).

4. Commit your changes:

```bash
git add train.py
git commit -m "Add validation channel and metrics output"
```

> "This mirrors a real development workflow: base script, branch, enhance, test, commit. In Module 3 you will use the validation metrics to decide whether a model should be approved for deployment."

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| `ModuleNotFoundError: sagemaker` | Install with `pip install sagemaker` in the notebook. |
| Training job stuck on "Starting" for 10+ minutes | Normal for first job in a new account/region. SageMaker is pulling the container for the first time. Subsequent jobs are faster. |
| `FileNotFoundError` during training | Verify the S3 path in the `.fit()` call matches where the data was uploaded. Check channel names match. |
| `ClientError: Access Denied` | The execution role lacks S3 permissions. Attach the appropriate policy in IAM. |
| Local Mode fails with "Docker not found" | Docker Desktop must be installed and running. In Studio, Docker is pre-installed. |
| Associates confuse `source_dir` with data path | `source_dir` is where the code lives. Data paths go in `.fit()`. They are separate S3 locations. |
| Artifact `model.tar.gz` is empty | The script did not save anything to `SM_MODEL_DIR`. Check the `joblib.dump()` path uses `model_dir` variable. |
