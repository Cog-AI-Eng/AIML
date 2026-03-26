# Guide 4: Monitor and Inspect the Training Job

While the notebook shows you streaming logs, the AWS console provides a richer view of the training job: its lifecycle stages, resource configuration, input/output paths, and detailed CloudWatch logs. This guide teaches you to navigate the console view of a training job.

---

## Steps

### Step 1 -- Find Your Training Job in the Console

1. Open the **SageMaker console** in a new browser tab (keep your Studio notebook tab open).
2. Go to **Training** -> **Training jobs** in the left navigation.
3. Find your training job in the list. It will be named something like `fraudshield-rf-2026-03-22-14-30-00-123`.
4. Click on the job name to open the details page.

### Step 2 -- Examine the Job Summary

At the top of the details page, note:

| Field | What to Look For |
|-------|-----------------|
| **Training job status** | Should be `Completed` (green) or `InProgress` (blue) |
| **Training duration** | Total wall-clock time |
| **Billable time** | Only the time the container was running |
| **Creation time** / **End time** | Full timestamp range |
| **ARN** | The unique Amazon Resource Name for this job |

### Step 3 -- Review the Resource Configuration

Scroll to the **Resource configuration** section:

1. **Instance type:** Confirm it is `ml.m5.xlarge`
2. **Instance count:** Should be `1`
3. **Volume size:** The attached EBS storage (default 30 GB)
4. These values match what you passed to the `SKLearn` Estimator in Guide 3.

### Step 4 -- Review the Input Data Configuration

Find the **Input data configuration** section:

1. **Channel name:** `train` -- this maps to `SM_CHANNEL_TRAIN` inside the container
2. **S3 data source:** The S3 URI you provided in `.fit()`
3. **S3 data type:** `S3Prefix` -- SageMaker downloaded everything under this prefix
4. **Input mode:** `File` -- data was downloaded to the container's local filesystem before training started

### Step 5 -- Review the Output Data Configuration

Find the **Output data configuration** section:

1. **S3 output path:** The base path where artifacts are stored
2. The complete artifact location follows the pattern:
   ```
   <S3 output path>/<training-job-name>/output/model.tar.gz
   ```
3. Click the S3 output path link if available -- it should take you to the S3 location.

### Step 6 -- Review Hyperparameters

Find the **Hyperparameters** section:

1. You should see the values you passed to the Estimator:
   - `n-estimators`: `100`
   - `random-state`: `42`
2. SageMaker may also show additional framework-injected parameters (e.g., `sagemaker_program`, `sagemaker_submit_directory`). These are how SageMaker tells the container where to find your script.

### Step 7 -- Open CloudWatch Logs

1. In the **Monitor** section (or look for a **View logs** link), click to open CloudWatch Logs.
2. This opens the CloudWatch console, filtered to the log group `/aws/sagemaker/TrainingJobs`.
3. Find the log stream for your training job (named after the job).
4. Click on it and read through the log entries. You should see:
   - Container startup messages
   - Data download progress
   - Your script's `print()` output (training accuracy, model saved message)
   - Training completion and artifact upload confirmation
5. This is the primary debugging tool when training jobs fail -- the full stdout/stderr from your script appears here.

### Step 8 -- Map the Seven Lifecycle Steps

Based on what you observed in the console and logs, identify the seven steps of a SageMaker training job:

| Step | What Happens | Where You See Evidence |
|------|-------------|----------------------|
| 1. Provisioning | EC2 instance allocated | Job status transitions from Creating to Training |
| 2. Pulling container | Docker image downloaded from ECR | Early CloudWatch log entries |
| 3. Downloading input data | S3 data copied to `/opt/ml/input/data/` | Log entries about data download |
| 4. Running training script | Your `train.py` executes | Your print statements in the logs |
| 5. Uploading model artifacts | `/opt/ml/model/` compressed and uploaded to S3 | Log entries about model upload |
| 6. Writing logs and metrics | stdout/stderr flushed to CloudWatch | The logs you are reading right now |
| 7. Teardown | Instance terminated and resources released | Job status changes to Completed |

---

## Presentation Checkpoint

Be prepared to show:
- The training job details page with status **Completed**
- The **Resource configuration** section confirming `ml.m5.xlarge`
- The **Input data configuration** showing your S3 data path and channel name
- The **CloudWatch Logs** with your training script's output
- Your mapping of the seven lifecycle steps to evidence in the console/logs
- Explain: What is the difference between "Training duration" and "Billable time"? (Training duration is wall-clock time including provisioning; billable time is only the time the container was actively running your code)

---

## Key Concepts

- **Training Job Lifecycle:** The seven steps from provisioning to teardown. Understanding this lifecycle is essential for debugging -- if a job fails at step 3 (data download), the problem is likely an S3 path or IAM permission; if it fails at step 4 (script execution), the problem is in your code.
- **CloudWatch Logs:** The primary debugging tool for SageMaker. Every `print()` statement and error traceback from your training script appears here.
- **Ephemeral Compute:** Training instances are automatically terminated after the job completes. You are only charged for billable seconds -- no cleanup required for the compute resources themselves.
