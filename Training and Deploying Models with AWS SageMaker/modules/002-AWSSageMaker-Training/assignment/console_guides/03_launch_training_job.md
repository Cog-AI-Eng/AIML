# Guide 3: Launch a Training Job from Studio

Now that you understand the console form fields and have data in S3, you will launch an actual training job. You will do this from a Studio notebook, using the SageMaker Python SDK. The code follows the exact pattern demonstrated in the lecture.

---

## Steps

### Step 1 -- Open a Studio Notebook

1. From the **SageMaker console**, go to **Domains** -> **fraudshield-domain**.
2. Click **Open Studio** next to your default user profile.
3. In Studio, click **File** -> **New** -> **Notebook**.
4. When prompted for a kernel/image, select a **Python 3 (Data Science)** kernel or similar. If you see options for instance type, select `ml.t3.medium` (the Free Tier notebook instance).
5. Wait for the kernel to start.

### Step 2 -- Write the Training Script

In the first notebook cell, create the training script that SageMaker will execute inside the container. This script follows the **Script Mode** structure from the readings:

```python
%%writefile train.py
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

    train_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(train_dir, f)) for f in train_files])

    # Drop transaction_id (not a feature) and separate target
    df = df.drop(columns=["transaction_id"], errors="ignore")
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    print(f"Training on {len(X)} samples, {X.shape[1]} features, "
          f"fraud rate: {y.mean():.1%}")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model.fit(X, y)
    print(f"Training accuracy: {model.score(X, y):.4f}")

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    print(f"Model saved to {model_dir}/model.pkl")

if __name__ == "__main__":
    main()
```

Run this cell. It creates a `train.py` file in the notebook's working directory.

### Step 3 -- Configure and Launch the Estimator

In the next cell, configure the SageMaker Estimator and launch the training job:

```python
import sagemaker
from sagemaker.sklearn import SKLearn

# Get the SageMaker session and default role
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = "fraudshield-training-data-<your-initials>"  # UPDATE THIS

# Configure the Estimator
estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    hyperparameters={
        "n-estimators": 100,
        "random-state": 42,
    },
    output_path=f"s3://{bucket}/output",
    base_job_name="fraudshield-rf",
)

# Launch the training job
estimator.fit({
    "train": f"s3://{bucket}/data/train/",
})
```

**Before running:** Update the `bucket` variable to match your actual bucket name.

Run this cell. The output will show the training job progress in real time.

### Step 4 -- Observe the Training Output

While the job runs, watch the notebook output. You will see:
1. **Job name** (e.g., `fraudshield-rf-2026-03-22-14-30-00-123`)
2. **Provisioning** messages
3. **Container download** progress
4. **Training logs** (stdout from your `train.py` script)
5. **Completion** message with billable seconds

Note the job name -- you will use it to find the job in the console in Guide 4.

### Step 5 -- Record Key Information

After the job completes, run this cell to get the key outputs:

```python
print(f"Training job name: {estimator.latest_training_job.name}")
print(f"Model artifact: {estimator.model_data}")
print(f"Billable seconds: {estimator.latest_training_job.describe()['BillableTimeInSeconds']}")
```

Write down:
- The training job name
- The model artifact S3 URI
- The billable seconds

---

## Presentation Checkpoint

Be prepared to show:
- The notebook with the training script and SDK code
- The training output showing the job progressing through its lifecycle stages
- The training job name and model artifact location
- Explain: What are the three structural requirements of a Script Mode training script? (Main guard `if __name__ == "__main__"`, argument parser for hyperparameters, SageMaker environment variables for data/model paths)
- Explain: What does `estimator.fit()` do behind the scenes? (Provisions an EC2 instance, pulls the framework container, downloads your script and training data from S3, runs the script, uploads the model artifacts to S3, and terminates the instance)

---

## Key Concepts

- **`%%writefile`:** A Jupyter magic command that writes the cell contents to a file. This is how you create the training script from within a notebook.
- **`SKLearn` Estimator:** A framework-specific Estimator class that pre-configures the correct scikit-learn container. You provide the script; SageMaker provides the runtime.
- **`.fit()`:** The method that launches the training job. It accepts a dictionary mapping channel names to S3 URIs.
- **Billable Seconds:** SageMaker only charges for the time the training container is actually running, not for provisioning or teardown.

---

## AIML Connection

The *Algorithm Selection Framework* reading discussed matching algorithms to data characteristics. Here, the `RandomForestClassifier` was chosen for tabular fraud data -- an ensemble of decision trees that handles mixed feature types well. The `n-estimators` hyperparameter controls the number of trees, and `random-state` ensures reproducibility.
