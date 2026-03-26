# Guide 2: Explore the Create Training Job Form

Before launching a training job from code, it is valuable to see every configuration option in the console. This guide walks you through the "Create training job" form field by field, mapping each one to its SageMaker SDK equivalent. You will NOT submit the form -- this is a mapping exercise.

---

## Steps

### Step 1 -- Navigate to the Training Job Form

1. In the **SageMaker console**, go to **Training** -> **Training jobs** in the left navigation.
2. Click **Create training job**.
3. You are now looking at the configuration form. Do NOT fill it in or submit it yet -- walk through each section and understand what it controls.

### Step 2 -- Job Settings Section

| Console Field | What It Controls | SDK Equivalent |
|--------------|-----------------|----------------|
| **Training job name** | Unique identifier for this job | `base_job_name` parameter on the Estimator (SDK auto-appends a timestamp) |
| **IAM role** | The execution role SageMaker assumes | `role` parameter on the Estimator |

Note: In the console, you provide the full job name. With the SDK, you provide a `base_job_name` and SageMaker appends a timestamp to ensure uniqueness.

### Step 3 -- Algorithm Section

| Console Field | What It Controls | SDK Equivalent |
|--------------|-----------------|----------------|
| **Algorithm source** | Built-in algorithm, your own script, or your own container | Determines which Estimator class you use (`SKLearn`, `PyTorch`, or generic `Estimator`) |
| **Algorithm / Training image** | The Docker container with the ML framework | `framework_version` on framework Estimators, or `image_uri` on the generic Estimator |
| **Script Mode / Entry point** | Your training script filename | `entry_point` parameter |
| **Source directory** | S3 location of your code | `source_dir` parameter |

This is where you choose between **Script Mode** (Bring Your Own Script) and **BYOM** (Bring Your Own Model/Container):
- **Script Mode:** You provide a `.py` script; SageMaker provides the container with the framework pre-installed
- **BYOM:** You provide a complete Docker image from ECR; SageMaker just provides the compute

### Step 4 -- Resource Configuration Section

| Console Field | What It Controls | SDK Equivalent |
|--------------|-----------------|----------------|
| **Instance type** | The EC2 instance type for training | `instance_type` (must be `ml.m5.xlarge` for this course) |
| **Instance count** | Number of training instances | `instance_count` (use `1` for single-instance training) |
| **Additional storage volume** | EBS volume attached to the instance | `volume_size` (default 30 GB) |
| **Maximum runtime** | Timeout to prevent runaway jobs | `max_run` (default 86400 seconds = 24 hours) |

### Step 5 -- Hyperparameters Section

| Console Field | What It Controls | SDK Equivalent |
|--------------|-----------------|----------------|
| **Key-value pairs** | Hyperparameters passed to your script | `hyperparameters` dict on the Estimator |

In the console, you add each hyperparameter as a key-value pair (e.g., key: `n-estimators`, value: `100`). Your Script Mode training script receives these via `argparse` command-line arguments.

### Step 6 -- Input Data Configuration Section

| Console Field | What It Controls | SDK Equivalent |
|--------------|-----------------|----------------|
| **Channel name** | Name of the input channel (e.g., `train`) | Key in the dict passed to `.fit()` |
| **S3 data location** | S3 URI for the training data | Value in the dict passed to `.fit()` |
| **S3 data type** | `S3Prefix` (folder) or `ManifestFile` | `s3_input` configuration |
| **Content type** | MIME type of the data (e.g., `text/csv`) | `content_type` on the `TrainingInput` |
| **Input mode** | `File` (download) or `Pipe` (stream) | `input_mode` on the `TrainingInput` |

Note: The channel name `train` maps to the environment variable `SM_CHANNEL_TRAIN` and the path `/opt/ml/input/data/train/` inside the container.

### Step 7 -- Output Data Configuration Section

| Console Field | What It Controls | SDK Equivalent |
|--------------|-----------------|----------------|
| **S3 output path** | Where SageMaker writes model.tar.gz | `output_path` on the Estimator |

After training, SageMaker compresses the contents of `/opt/ml/model/` into `model.tar.gz` and uploads it to:
```
<output_path>/<training-job-name>/output/model.tar.gz
```

### Step 8 -- Document Your Mapping

Before leaving this page, document the complete mapping. For each field you explored, write down:
1. The console field name
2. What you would enter for the FraudShield training job
3. The equivalent SDK parameter

Then click **Cancel** -- do not submit the form. You will launch the job from a notebook in Guide 3.

---

## Presentation Checkpoint

Be prepared to show:
- Your completed console-to-SDK mapping document
- Explain: What is the difference between the "Algorithm source" options? (Script Mode uses a SageMaker-managed container with your script injected; BYOM uses your own Docker image from ECR)
- Explain: How does a channel name in the "Input data configuration" end up inside the training container? (SageMaker maps it to an environment variable `SM_CHANNEL_<NAME>` and downloads the data to `/opt/ml/input/data/<name>/`)
- Explain: Where will the trained model artifact end up in S3? (At `<output_path>/<job_name>/output/model.tar.gz`)

---

## Key Concepts

- **Console-to-SDK Mapping:** Every console field has a direct SDK equivalent. Understanding this mapping helps you move between the console (for exploration) and the SDK (for automation).
- **Input Channels:** Named data sources that SageMaker downloads into the container. Standard channels are `train` and `validation`.
- **Script Mode vs BYOM:** Script Mode is recommended for most use cases -- you write a training script and SageMaker provides the runtime. BYOM is for advanced cases where you need full container control.
