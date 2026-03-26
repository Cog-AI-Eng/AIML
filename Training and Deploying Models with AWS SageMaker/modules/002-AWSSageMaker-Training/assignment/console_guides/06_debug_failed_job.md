# Guide 6: Debug a Failed Training Job

In real projects, training jobs fail regularly -- wrong S3 paths, missing permissions, broken scripts, timeout limits. Knowing how to diagnose failures quickly is essential. In this guide, you will intentionally launch a job that fails and practice the debugging workflow.

---

## Steps

### Step 1 -- Launch a Job with a Wrong S3 Path

Return to your Studio notebook and run a new cell with an intentionally incorrect S3 path:

```python
# Intentional failure: wrong S3 path
debug_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    hyperparameters={
        "n-estimators": 50,
        "random-state": 42,
    },
    output_path=f"s3://{bucket}/output",
    base_job_name="fraudshield-rf-debug",
)

# Wrong path: "data/nonexistent/" does not exist
debug_estimator.fit({
    "train": f"s3://{bucket}/data/nonexistent/",
})
```

The job will start provisioning and then fail. You may see an error in the notebook output, or the cell may hang and eventually report a failure.

### Step 2 -- Find the Failed Job in the Console

1. Go to **SageMaker** -> **Training** -> **Training jobs**.
2. Find the `fraudshield-rf-debug-*` job. Its status will show **Failed** (red).
3. Click on the job name to open the details page.

### Step 3 -- Read the Failure Reason

1. At the top of the details page, you will see a **Failure reason** field.
2. The message will indicate what went wrong. For a wrong S3 path, you might see something like:
   - `"ClientError: Data download failed..."` or
   - `"Could not find the specified channel/path..."`
3. This high-level message tells you *where* in the lifecycle the job failed (data download = Step 3 of the seven-step lifecycle).

### Step 4 -- Check CloudWatch Logs for Details

1. Click the **View logs** link (or navigate to **CloudWatch** -> **Log groups** -> `/aws/sagemaker/TrainingJobs`).
2. Find the log stream for your failed job.
3. Read through the entries. Look for:
   - The exact error message and traceback
   - Which step the container was on when it failed
   - Any permission-related errors (AccessDenied) vs. path-related errors (FileNotFoundError)

### Step 5 -- Diagnose the Common Failure Patterns

Use this reference to categorize failures:

| Error Pattern | Category | Likely Cause | Fix |
|--------------|----------|-------------|-----|
| `AccessDenied` or `Access Denied` | IAM Permissions | Execution role lacks S3 permissions for this bucket | Update the IAM role's S3 policy to include the correct bucket ARN |
| `FileNotFoundError` or `No such file` | Wrong S3 Path | The data does not exist at the specified S3 prefix | Verify the S3 URI in the `.fit()` call matches the actual data location |
| `ModuleNotFoundError` | Missing Dependency | A Python package your script imports is not in the container | Add a `requirements.txt` to your `source_dir` listing the missing package |
| `ResourceLimitExceeded` | Service Limit | Your account has hit the instance limit for this type | Request a limit increase through AWS Support or use a different instance type |
| Timeout | Max Runtime | The job exceeded `max_run` seconds | Increase `max_run`, reduce data size, or optimize your script |

### Step 6 -- Compare Successful vs. Failed Jobs

Open both training jobs in the console (or switch between tabs):

| Aspect | Successful Job | Failed Job |
|--------|---------------|------------|
| **Status** | Completed (green) | Failed (red) |
| **Training duration** | Full duration | Partial (failed during data download) |
| **Billable seconds** | Full billable time | Reduced (you pay for the time before failure) |
| **Output artifacts** | `model.tar.gz` in S3 | No artifacts (job never reached Step 5) |
| **CloudWatch Logs** | Training output and success messages | Error traceback and failure messages |

### Step 7 -- (Optional) Trigger Other Failure Types

If time permits, try these additional failure scenarios:

**Permission failure:** Modify the code to use a different S3 bucket that your role does not have access to:
```python
# Will fail with AccessDenied if role cannot access this bucket
debug_estimator.fit({"train": "s3://some-other-bucket/data/train/"})
```

**Script failure:** Modify `train.py` to import a package that is not installed (e.g., `import xgboost`), then re-run.

---

## Presentation Checkpoint

Be prepared to show:
- The **failed** training job in the console with its **Failed** status
- The **Failure reason** field and what it tells you
- The **CloudWatch Logs** for the failed job, highlighting the key error message
- Your diagnosis: which of the seven lifecycle steps did the job fail at, and why?
- The side-by-side comparison between your successful and failed jobs
- Explain: If you saw `AccessDenied` instead of `FileNotFoundError`, what would that tell you? (The S3 path might be correct, but the IAM role does not have permission to access that bucket. You would need to update the role's policy.)
- Explain: Why are CloudWatch Logs the primary debugging tool for training jobs? (The training container runs on a managed instance you cannot SSH into. CloudWatch is the only way to see stdout/stderr from your script.)

---

## Key Concepts

- **Failure Diagnosis Workflow:** Check the console's failure reason first (high-level category), then dive into CloudWatch Logs for the full traceback and context.
- **Lifecycle-Based Debugging:** Knowing *which step* failed tells you where to look. Data download failures point to S3/IAM issues. Script execution failures point to code bugs. Artifact upload failures point to output path problems.
- **Billable on Failure:** You are still charged for the compute time before the failure. This is why fast failure detection (wrong paths fail during data download, not during training) is important.
