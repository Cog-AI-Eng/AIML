# Estimators & Configurations

**Estimated Time:** 10 Minutes

## Introduction

In the *Script Mode Structure* reading you learned what a SageMaker training script looks like on the inside: a main guard, an argument parser, and standard directory paths. But a script by itself does not run -- it needs infrastructure. Someone has to tell SageMaker which script to run, on what instance type, with what data, and with what hyperparameters. That "someone" is either you clicking through the SageMaker console or an **Estimator** object in the SDK that bundles all of those decisions into a single configuration.

If you used scikit-learn in the AIML Foundations module, you already know the pattern. A `RandomForestClassifier(n_estimators=100, random_state=42)` is an estimator: it packages algorithm choice and hyperparameters into one object, and you call `.fit(X, y)` to run training. A SageMaker Estimator works the same way, except instead of `.fit(X, y)` running on your laptop, `.fit()` launches a managed training job on AWS infrastructure, reads data from S3, and writes model artifacts back to S3.

This reading walks through every configuration parameter -- first in the console UI so you see each field, then in the SDK so you can automate the same thing in code.

## Core Concepts

### Creating a training job in the console

The SageMaker console exposes every training job parameter as a form field. Walking through this form is the best way to understand what an Estimator configures, because every SDK parameter maps directly to one of these fields.

1. **Navigate to Training Jobs.** In the SageMaker console sidebar, click **Training > Training jobs**, then click **Create training job**.

2. **Job settings section:**
   - **Job name:** A unique identifier for this training run (e.g., `rf-classifier-2026-03-22`). SageMaker uses this name in CloudWatch logs and in the training job list. Choose descriptive names so you can find past jobs easily.
   - **IAM role:** The execution role that grants the training job permission to access S3, CloudWatch, and other services. Select the least-privilege role you created in the *IAM* reading, or choose the default Domain role for learning exercises.

3. **Algorithm source section:**
   - **Algorithm options:** Select **SageMaker built-in algorithm** and choose a framework (e.g., "Scikit Learn 1.2-1"). This tells SageMaker which managed container to use.
   - **Entry point:** The filename of your Script Mode training script (e.g., `train.py`).
   - **Source directory:** The S3 URI where your script files are stored.

4. **Resource configuration section:**
   - **Instance type:** The compute instance that will run your training job. For this curriculum, always use `ml.m5.xlarge` to stay within Free Tier limits. This field is where cost control happens -- a `ml.p3.2xlarge` GPU instance costs roughly 10x more per hour.
   - **Instance count:** How many instances to use. For single-machine training (which covers most scikit-learn and small PyTorch workloads), set this to `1`. Values greater than 1 enable distributed training, which is beyond the scope of this curriculum.
   - **Volume size:** The amount of disk storage (in GB) attached to the training instance. The default (30 GB) is sufficient for most workloads. Increase it only if your dataset or intermediate files exceed that.
   - **Max runtime:** A safety cutoff in seconds. If your training job runs longer than this, SageMaker terminates it. Set this to prevent runaway jobs from consuming resources indefinitely. A reasonable default for learning exercises is 3600 seconds (1 hour).

5. **Hyperparameters section:**
   - Click **Add hyperparameter** to define key-value pairs. Each key must match an argument in your script's `argparse` parser. For example:
     - Key: `n-estimators`, Value: `100`
     - Key: `random-state`, Value: `42`
   - SageMaker passes these as command-line arguments to your script.

6. **Input data configuration section:**
   - **Channel name:** The name of the data channel (e.g., `train`, `validation`). This maps to `SM_CHANNEL_TRAIN` and `SM_CHANNEL_VALIDATION` in your script.
   - **S3 data location:** The S3 URI where your training data lives (e.g., `s3://my-bucket/data/train/`).
   - **S3 data type:** Choose `S3Prefix` for a directory of files or `ManifestFile` for a manifest listing specific files.
   - **Input mode:** `File` downloads the full dataset before training starts (most common). `Pipe` streams data directly, which is useful for very large datasets.
   - You can add multiple channels by clicking **Add channel** (e.g., one for training data, one for validation data).

7. **Output data configuration section:**
   - **S3 output path:** The S3 URI where SageMaker will upload the `model.tar.gz` artifact after training (e.g., `s3://my-bucket/output/`).

8. **Click Create training job.** SageMaker provisions the instance, downloads the data, runs your script, uploads the model artifact, and tears down the instance. You can monitor progress under **Training > Training jobs** in the sidebar.

> **Tip:** After the job completes, click its name to see the full configuration, metrics, logs, and output path. This details page is invaluable for debugging.

### The SDK Estimator: same parameters, one object

Every field on the console form maps to a parameter on the SDK Estimator. Here is the equivalent for the console walkthrough above:

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point="train.py",
    source_dir="code/",
    role="arn:aws:iam::123456789012:role/SageMaker-Training-LeastPrivilege",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size=30,
    max_run=3600,
    framework_version="1.2-1",
    hyperparameters={
        "n-estimators": 100,
        "random-state": 42,
    },
    output_path="s3://my-bucket/output/",
)
```

And the `.fit()` call maps to the input data configuration:

```python
estimator.fit({
    "train": "s3://my-bucket/data/train/",
    "validation": "s3://my-bucket/data/validation/",
})
```

The dictionary keys (`"train"`, `"validation"`) are the channel names. Calling `.fit()` submits the training job to SageMaker -- exactly what clicking "Create training job" does in the console.

### Parameter-to-field mapping

| Console Field | SDK Parameter | What It Controls |
| :--- | :--- | :--- |
| Job name | Auto-generated (or `base_job_name`) | Unique job identifier |
| IAM role | `role` | Permissions for the training job |
| Algorithm / Framework | Estimator class (e.g., `SKLearn`, `PyTorch`) | Which managed container to use |
| Entry point | `entry_point` | Your training script filename |
| Source directory | `source_dir` | Where your script files are stored |
| Instance type | `instance_type` | Compute hardware |
| Instance count | `instance_count` | Number of machines |
| Volume size | `volume_size` | Disk storage in GB |
| Max runtime | `max_run` | Safety timeout in seconds |
| Hyperparameters | `hyperparameters` | Key-value pairs passed to your script |
| Channel name + S3 path | `.fit({"channel": "s3://..."})` | Training data locations |
| S3 output path | `output_path` | Where model artifacts are uploaded |

This table is worth bookmarking. When you are debugging a training job, you can trace any issue back to a specific parameter on either side.

### Framework estimators vs. generic estimators

The SageMaker SDK provides framework-specific estimator classes that simplify Script Mode configuration:

- `sagemaker.sklearn.SKLearn` -- scikit-learn
- `sagemaker.pytorch.PyTorch` -- PyTorch
- `sagemaker.tensorflow.TensorFlow` -- TensorFlow
- `sagemaker.xgboost.XGBoost` -- XGBoost

Each class knows which managed container to pull and automatically sets up the Script Mode execution environment. You only need to specify the `framework_version`.

The generic `sagemaker.estimator.Estimator` class is used when you bring your own container (BYOM). Instead of a framework version, you pass an `image_uri` pointing to your ECR image. As discussed in the *BYOM & BYOS Approaches* reading, this is the advanced path.

### Local mode for fast iteration

During development, waiting several minutes for SageMaker to provision an instance for every training run slows you down. **Local mode** lets you test your Script Mode script on your own machine (or inside a Studio notebook instance) using Docker, without submitting a job to SageMaker infrastructure.

To use local mode, set `instance_type` to `"local"`:

```python
estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="local",
    framework_version="1.2-1",
    hyperparameters={"n-estimators": 50},
)
estimator.fit({"train": "file://./data/train/"})
```

Notice the `"file://"` prefix on the data path -- local mode reads from your local filesystem instead of S3. This lets you iterate rapidly on script bugs and logic errors before committing to a full cloud training job. Switch `instance_type` back to `"ml.m5.xlarge"` and the S3 path back to `"s3://..."` when you are ready for the real run.

## Connecting to Practice

This reading gives you the complete parameter reference for SageMaker training jobs. In the *Estimators & Configurations Video*, you will see a live walkthrough of both the console form and the SDK code. The next reading, *Training Job Anatomy*, explains what happens behind the scenes after you click "Create" or call `.fit()`. And in the module assignment, you will configure and launch your own training job end to end.

The most useful thing you can do right now is open the SageMaker console, navigate to **Training > Training jobs > Create training job**, and click through each section without submitting. Match every field to the parameter mapping table above. This dry run builds muscle memory so that when you write your first Estimator in code, every parameter will feel familiar.

---

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Python SDK - Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)** - *Docs*: The full API reference for all Estimator classes, including every parameter and its default value.
- **[SageMaker Instance Types and Pricing](https://aws.amazon.com/sagemaker/pricing/)** - *Docs*: A detailed breakdown of instance types available for training, with on-demand pricing so you can make cost-conscious choices.

**Interactive practice**

- **[SageMaker Examples - Training](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk)** - *Interactive*: Runnable notebook examples for scikit-learn, PyTorch, TensorFlow, and XGBoost training jobs with full Estimator configurations.
- **[AWS Hands-On: Train a Model](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-train-a-model/)** - *Interactive*: A free guided lab that walks you through configuring and running a training job in your own console.
