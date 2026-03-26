# Script Mode Structure

**Estimated Time:** 10 Minutes

## Introduction

In the *BYOM & BYOS Approaches* reading you learned that Script Mode is the recommended way to bring your own training code to SageMaker. You supply a Python script; SageMaker supplies the container, the compute, and the plumbing that connects your data in S3 to your code and your trained model back to S3. But Script Mode is not magic -- your script has to follow a specific structure so SageMaker knows how to run it.

If you wrote training scripts in the AIML Foundations module, you probably had something like: load a CSV from a local path, parse some hyperparameters, fit a model, and save it with `joblib.dump()`. A Script Mode script does the same things, but the *where* changes. Data arrives in a container directory that SageMaker populates from S3. Hyperparameters arrive as command-line arguments that SageMaker passes automatically. And the trained model must be saved to a specific output directory so SageMaker can package it and upload it back to S3.

This reading breaks down the three structural requirements every Script Mode script must meet: the **main guard**, the **argument parser**, and the **standard directory paths**. By the end, you will be able to take a local training script and adapt it for SageMaker.

## Core Concepts

### The container directory layout

When SageMaker starts a training job, it provisions a compute instance, pulls the managed framework container, and mounts a standard directory tree inside it. Your script runs inside this tree and reads/writes from specific paths:

```
/opt/ml/
├── input/
│   ├── config/
│   │   ├── hyperparameters.json
│   │   └── resourceconfig.json
│   └── data/
│       ├── train/        <-- your training data (downloaded from S3)
│       └── validation/   <-- your validation data (if provided)
├── model/                <-- save your trained model here
└── output/
    └── failure           <-- write error messages here if training fails
```

**`/opt/ml/input/data/<channel_name>/`** is where SageMaker downloads your training data before your script starts. The channel name (like `train` or `validation`) matches what you specify when you configure the training job. If you set up a channel called `train` pointing to `s3://my-bucket/data/train/`, SageMaker downloads that S3 prefix into `/opt/ml/input/data/train/`.

**`/opt/ml/model/`** is where you save your trained model file. After your script finishes, SageMaker compresses everything in this directory into a `model.tar.gz` artifact and uploads it to the S3 output path you specified. This is the SageMaker equivalent of your local `joblib.dump(model, 'model.pkl')` -- you just save to a different directory.

**`/opt/ml/input/config/hyperparameters.json`** contains the hyperparameters you passed to the training job, serialized as JSON. You do not need to read this file directly because SageMaker also passes hyperparameters as command-line arguments to your script.

### From local script to Script Mode script

Here is a side-by-side comparison showing how a typical local training script maps to a Script Mode script.

**Local script (what you wrote in AIML Foundations):**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("data/train.csv")
X = data.drop("target", axis=1)
y = data["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
```

**Script Mode equivalent:**

```python
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))


if __name__ == "__main__":
    main()
```

The logic is identical. Three things changed: how data is located, how hyperparameters are received, and where the model is saved. Let us examine each requirement.

### Requirement 1: The main guard

Every Script Mode script must include:

```python
if __name__ == "__main__":
    main()
```

This is standard Python practice, but it is mandatory here because SageMaker executes your script as a module. Without the guard, your training code would run during import, which causes errors in SageMaker's execution framework. Wrap your training logic in a function (conventionally called `main()`) and call it from the guard.

### Requirement 2: The argument parser

Hyperparameters are passed to your script as command-line arguments. SageMaker reads them from the training job configuration and appends them to the script invocation. Your script must use `argparse` (or a compatible parser) to receive them:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=0.01)
args = parser.parse_args()
```

The argument names you define here must match the hyperparameter keys you pass when creating the training job. SageMaker converts underscores and hyphens interchangeably, so `--n-estimators` in the parser matches `n_estimators` or `n-estimators` in the job configuration.

Using an argument parser instead of hardcoded values means you can run the same script with different hyperparameters without editing code -- a direct application of the reproducibility principles from the AIML Foundations module.

### Requirement 3: Standard directory paths via environment variables

SageMaker sets environment variables inside the container that tell your script where to find data and where to save the model. The most important ones:

| Environment Variable | Points To | Purpose |
| :--- | :--- | :--- |
| `SM_CHANNEL_TRAIN` | `/opt/ml/input/data/train` | Directory containing your training data |
| `SM_CHANNEL_VALIDATION` | `/opt/ml/input/data/validation` | Directory containing your validation data |
| `SM_MODEL_DIR` | `/opt/ml/model` | Directory where you must save your trained model |
| `SM_OUTPUT_DATA_DIR` | `/opt/ml/output/data` | Directory for additional output files (plots, logs) |
| `SM_NUM_GPUS` | Number of GPUs available | Useful for distributed training configuration |

Read these with `os.environ.get()` and provide fallback defaults so you can test the script locally too:

```python
train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
```

The fallback defaults let you run the same script on your laptop (where the environment variables do not exist) and on SageMaker (where they do). This dual-mode capability is valuable during development.

### Where the entry point is specified in the console

When you create a training job in the SageMaker console, the entry point configuration appears after you select a framework algorithm:

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. Under **Algorithm source**, select the framework (e.g., "Scikit Learn").
3. The form reveals fields for:
   - **Entry point:** The filename of your script (e.g., `train.py`).
   - **Source directory:** The S3 URI where your script is stored (e.g., `s3://my-bucket/code/`). SageMaker downloads this directory and executes the entry point from it.
   - **Dependencies:** An optional file listing additional Python packages to install before running your script.

These three fields tell SageMaker: "download the code from this S3 location, install any extra dependencies, and run this specific Python file."

### SDK equivalent

In the SDK, the same information is passed through the framework estimator constructor:

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point="train.py",
    source_dir="code/",
    role=role_arn,
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
    hyperparameters={
        "n-estimators": 100,
        "random-state": 42,
    },
)
```

The `entry_point`, `source_dir`, and `hyperparameters` parameters map directly to the console form fields. The *Estimators & Configurations* topic covers the full estimator API.

## Connecting to Practice

This reading gives you the blueprint for every training script you will write in this curriculum. In the *Script Mode Structure Video*, you will see a live walkthrough of converting a local script to Script Mode. In the *Estimators & Configurations* reading, you will learn how to configure the SDK to run your script. And in the module assignment, you will write and execute a complete Script Mode training job.

The most useful thing you can do right now is take a training script from your AIML Foundations work, open it side by side with the Script Mode example above, and identify the three changes you would need to make: argument parser for hyperparameters, environment variables for data paths, and `SM_MODEL_DIR` for model output. That mental mapping is the foundation for everything in Module 2.

---

## Further Learning & Resources

**Documentation and reading**

- **[Using Scikit-learn with SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html)** - *Docs*: The SageMaker Python SDK guide for scikit-learn Script Mode, including entry point requirements and environment variable reference.
- **[SageMaker Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)** - *Docs*: The open-source toolkit that powers Script Mode's container-side execution, useful for understanding what happens between SageMaker launching the container and your script running.

**Interactive practice**

- **[SageMaker Examples - Scikit-learn Script Mode](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk/scikit_learn_iris)** - *Interactive*: A runnable notebook demonstrating a complete Script Mode workflow with scikit-learn on the Iris dataset.
- **[AWS Hands-On: Train a Model](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-train-a-model/)** - *Interactive*: A free guided lab covering the end-to-end Script Mode training flow in your own console.
