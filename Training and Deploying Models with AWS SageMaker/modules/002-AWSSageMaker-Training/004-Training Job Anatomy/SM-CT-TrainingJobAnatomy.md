# Training Job Anatomy

**Estimated Time:** 10 Minutes

## Introduction

In the *Estimators & Configurations* reading you learned how to fill out the training job form in the console and how to configure an SDK Estimator. You clicked "Create training job" (or called `.fit()`) and SageMaker started doing things. But what exactly happens between that moment and the point when a trained model artifact appears in S3?

Understanding the anatomy of a training job matters for the same reason understanding the ML lifecycle matters: when something goes wrong -- a permission error, a data loading failure, a script crash -- you need to know *where* in the process it broke. A training job is not a black box. It is a well-defined sequence of steps that you can observe and debug through the SageMaker console and CloudWatch logs.

This reading takes you inside a training job: the provisioning sequence, the container lifecycle, how input channels deliver data, how your script runs, and how outputs get back to S3. If you think of the *Estimators* reading as "how to order a meal," this reading is "what happens in the kitchen."

## Core Concepts

### The training job lifecycle

When SageMaker receives a training job request, it executes these steps in order:

**1. Provisioning.** SageMaker allocates the compute instance(s) you requested (e.g., one `ml.m5.xlarge`). This includes attaching the disk volume and configuring the network within your VPC. Provisioning typically takes 2-5 minutes. During this time, the job status in the console shows **Starting**.

**2. Pulling the container.** SageMaker pulls the Docker container image for your chosen framework (e.g., the scikit-learn 1.2-1 managed container) from Amazon ECR. If you are using BYOM, it pulls your custom image instead. The container is the runtime environment for your script.

**3. Downloading input data.** SageMaker reads the input data channels you configured and downloads the data from S3 into the container's `/opt/ml/input/data/<channel_name>/` directories. If you specified a `train` channel pointing to `s3://my-bucket/data/train/`, SageMaker copies those files into `/opt/ml/input/data/train/` inside the container. This happens before your script starts.

If you recall the *Train, Validation & Test Splits* topic from the AIML Evaluation module, you understand why separate splits matter. SageMaker channels are how you deliver those splits to the training job: one channel for training data, another for validation data. Your script reads from the appropriate directory without needing to know the S3 paths directly.

**4. Injecting and running your script.** For Script Mode, SageMaker downloads your entry point script (and source directory) from S3, places it in the container, and executes it. Hyperparameters from the job configuration are passed as command-line arguments. Environment variables like `SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`, and `SM_NUM_GPUS` are set before your script starts. Your script runs as a normal Python process inside the container.

**5. Uploading model artifacts.** When your script finishes (exits with code 0), SageMaker compresses everything in `/opt/ml/model/` into a `model.tar.gz` file and uploads it to the S3 output path you specified. This artifact is the trained model that later stages (deployment, registration) will use.

**6. Writing logs and metrics.** Throughout execution, SageMaker streams your script's `stdout` and `stderr` to **Amazon CloudWatch Logs**. Any `print()` statements, logging output, or error tracebacks appear there. SageMaker also captures built-in metrics (CPU utilization, memory usage, disk I/O) and any custom metrics your script emits.

**7. Teardown.** After the artifact is uploaded and logs are flushed, SageMaker terminates the instance and releases the resources. You are only charged for the time between provisioning and teardown. The job status changes to **Completed** (or **Failed** if the script exited with a non-zero code).

### The container's internal view

Here is a summary of what the container looks like from your script's perspective at the moment it starts running:

```
/opt/ml/
├── input/
│   ├── config/
│   │   ├── hyperparameters.json    <-- your hyperparameters as JSON
│   │   ├── inputdataconfig.json    <-- channel definitions
│   │   └── resourceconfig.json     <-- instance count, hostname info
│   └── data/
│       ├── train/                  <-- training data (downloaded from S3)
│       └── validation/             <-- validation data (if channel exists)
├── model/                          <-- EMPTY; your script writes here
├── output/
│   ├── data/                       <-- optional additional outputs
│   └── failure                     <-- write error message here on failure
└── code/                           <-- your entry point script
```

The `/opt/ml/input/config/` directory contains metadata files that SageMaker generates. Most Script Mode scripts do not read these directly because the SDK and environment variables provide the same information in a more convenient form. But they are useful for debugging: if your script cannot find its data, checking `inputdataconfig.json` tells you exactly which channels were configured and where they point.

### Inspecting a training job in the console

After a training job completes (or fails), the SageMaker console provides a detailed view. Here is how to navigate it:

1. **Navigate to the job.** In the SageMaker sidebar, click **Training > Training jobs**. You see a list of all jobs with their name, status, creation time, and duration. Click a job name to open its details page.

2. **Job summary section.** At the top of the details page you see:
   - **Status:** Completed, Failed, Stopped, or InProgress.
   - **Training duration:** How long the job ran (billable time).
   - **Training image:** The ECR URI of the container that was used.
   - **IAM role ARN:** The execution role the job ran under.

3. **Input data configuration.** This section lists each channel with its S3 URI, data type, and input mode. Use this to verify that the job read from the correct S3 location. If your script produced unexpected results, check here first -- a wrong S3 path is one of the most common errors.

4. **Output data configuration.** Shows the S3 URI where the `model.tar.gz` was uploaded. Click the S3 link to navigate directly to the artifact in the S3 console.

5. **Hyperparameters.** Lists every key-value pair that was passed to your script. Verify these match what you intended -- a misspelled key or wrong value type is another common debugging target.

6. **Monitoring section.** Shows instance-level metrics: CPU utilization, memory utilization, disk utilization, and GPU utilization (if applicable). These help you determine if your instance type is appropriately sized. If CPU is at 100% for the entire job, you might benefit from a larger instance. If it is at 10%, you might save money with a smaller one.

7. **View logs link.** Clicking this opens the CloudWatch Logs console, filtered to your training job's log stream. Here you see everything your script printed to `stdout` and `stderr`. Error tracebacks, `print()` debugging output, and framework-level messages all appear here in chronological order.

> **Tip:** You can also reach CloudWatch Logs directly: search "CloudWatch" in the console, navigate to **Log groups**, and look for `/aws/sagemaker/TrainingJobs`. Each training job creates its own log stream within this group.

### Handling failures

When a training job fails, the status shows **Failed** and the console displays an error message. Common failure patterns:

| Symptom | Likely Cause | Where to Look |
| :--- | :--- | :--- |
| "ClientError: Access Denied" | Execution role lacks S3 or SageMaker permissions | IAM role policies; check the role in the IAM console |
| "FileNotFoundError" in logs | Wrong S3 path or missing channel | Input data configuration on the job details page |
| "ModuleNotFoundError" in logs | Missing Python dependency in the container | Add a `requirements.txt` to your source directory |
| Script exits with non-zero code | Bug in your training script | CloudWatch logs; read the full traceback |
| Job times out (exceeds max runtime) | Training takes longer than expected | Increase `max_run` or reduce dataset size |

The `/opt/ml/output/failure` file is also relevant here: if your script catches an exception and writes a message to this file, SageMaker includes it in the failure reason displayed on the console.

### Early stopping and checkpointing (advanced)

For longer training jobs (especially deep learning), two features help you avoid wasting compute:

**Early stopping** monitors a validation metric during training and stops the job when the metric stops improving. In the SDK, you configure this on the Estimator or through a hyperparameter tuning job's `early_stopping_type` parameter. The concept is the same as the *Early Stopping Logic* topic from the AIML Evaluation module -- the difference is that SageMaker manages the stopping decision at the infrastructure level rather than inside your script's loop.

**Checkpointing** saves model snapshots at regular intervals to S3. If a training job is interrupted (spot instance reclamation, timeout), you can resume from the last checkpoint instead of starting over. You configure checkpointing by setting `checkpoint_s3_uri` and `checkpoint_local_path` on the Estimator. Your script must save checkpoint files to the local path; SageMaker syncs them to S3 in the background.

Both features are preferred-level topics in this curriculum and will be explored in the module assignment.

### SDK perspective

From the SDK, much of the training job anatomy is transparent: `.fit()` handles provisioning, data download, execution, and artifact upload. But you can observe the process programmatically:

```python
estimator.fit({"train": "s3://my-bucket/data/train/"}, wait=True, logs="All")
```

Setting `logs="All"` streams the CloudWatch logs directly into your notebook output during training. Setting `wait=True` (the default) blocks until the job completes. After completion, `estimator.model_data` returns the S3 path to the model artifact.

## Connecting to Practice

This reading gives you the mental model for what happens inside a training job. In the *Training Job Anatomy Video*, you will see a live demonstration of a job running, failing, and being debugged through the console. In the next reading, *Model Artifacts & S3 Storage*, you will learn what happens to the `model.tar.gz` after it lands in S3. And in the module assignment, you will run training jobs, inspect their output, and debug failures.

The most useful thing you can do right now is run a training job (either from the console or the SDK) and then explore every section of the job details page. Click into the CloudWatch logs and read the full output. Check the input and output S3 paths. This hands-on inspection will make every future debugging session faster.

---

## Further Learning & Resources

**Documentation and reading**

- **[How Training Works in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)** - *Docs*: The official deep dive into the training job lifecycle, container contract, and S3 data flow.
- **[SageMaker Training Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)** - *Docs*: AWS guidance on instance selection, data loading strategies, and cost optimization for training jobs.

**Interactive practice**

- **[Amazon CloudWatch Logs - Getting Started](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/WhatIsCloudWatchLogs.html)** - *Interactive*: Learn to navigate and query CloudWatch Logs, which is essential for debugging SageMaker training jobs.
- **[SageMaker Examples - Debugging Training Jobs](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-debugger)** - *Interactive*: Runnable notebooks demonstrating SageMaker Debugger for inspecting tensors and training anomalies.
