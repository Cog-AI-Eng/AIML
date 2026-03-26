# Guide 4: Build a Reproducibility Report

This guide walks you through assembling a reproducibility report for the best FraudShield model using information gathered from SageMaker Experiments and the Lineage Graph. The report documents everything another team member would need to recreate the exact same model from scratch.

---

## Steps

### Step 1 -- Identify the Best Model Run
1. In SageMaker Studio, open **Experiments** and click `fraudshield-fraud-detection`.
2. Sort the runs by `validation:auc` in descending order.
3. Click the best-performing run to open its detail page.
4. Note the run name and training job name for the report header.

### Step 2 -- Document the Data Version
1. From the run detail page, locate the **Input data** section (or navigate via the Lineage Graph to the Dataset artifact nodes).
2. Record the following in your report:
   - **Training data S3 URI:** the exact path (e.g., `s3://fraudshield-data-<account-id>/processed/train/`)
   - **Validation data S3 URI:** the exact path
   - **Data snapshot date:** check the S3 object's Last Modified timestamp in the S3 console
3. If your data is versioned (S3 versioning enabled), also record the S3 version ID for each file.
4. Note the number of training and validation records (from the training job logs or Data Wrangler profile).

### Step 3 -- Document the Hyperparameters
1. From the run detail page, open the **Parameters** or **Hyperparameters** tab.
2. Record every hyperparameter and its value in the report:
   - `objective`: (e.g., `binary:logistic`)
   - `eval_metric`: (e.g., `auc`)
   - `num_round`: (e.g., `150`)
   - `max_depth`: (e.g., `5`)
   - `eta`: (e.g., `0.2`)
   - `subsample`: (e.g., `0.7`)
3. Include any other parameters that were set, even if they used default values. Default values can change across algorithm versions.

### Step 4 -- Document the Training Environment
1. From the training job detail page (accessible from the run or from **Training > Training jobs** in the console), record:
   - **Algorithm:** XGBoost (note the exact version, e.g., `1.5-1`)
   - **Container image URI:** found in the training job details under Algorithm or Container
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`
   - **Volume size:** `10` GB
   - **IAM role ARN:** the execution role used
2. Also note the AWS region: `us-east-1`.

### Step 5 -- Document the Model Artifact
1. From the training job detail page, scroll to the **Output** section.
2. Record the model artifact location:
   - **Model artifact S3 URI:** (e.g., `s3://fraudshield-data-<account-id>/models/experiments/run2/output/model.tar.gz`)
3. Optionally record the S3 object ETag or version ID for verification.
4. Note the training job ARN for cross-referencing.

### Step 6 -- Document the Metrics
1. Record the final metric values from the run or training job:
   - **Final train:auc:** (e.g., `0.98`)
   - **Final validation:auc:** (e.g., `0.95`)
   - **Training duration:** (e.g., `247 seconds`)
   - **Billable seconds:** (from the training job detail page)
2. Note whether any early stopping was applied.

### Step 7 -- Compile the Report
1. Open a text editor or a Studio notebook and compile all documented fields into a structured report using this template:

   ```
   REPRODUCIBILITY REPORT
   ======================
   Model: FraudShield Fraud Detector
   Experiment: fraudshield-fraud-detection
   Run: [run name]
   Date: [today's date]

   DATA
   ----
   Training data: [S3 URI]
   Validation data: [S3 URI]
   Data snapshot date: [date]
   Training records: [count]
   Validation records: [count]

   HYPERPARAMETERS
   ---------------
   objective: [value]
   eval_metric: [value]
   num_round: [value]
   max_depth: [value]
   eta: [value]
   subsample: [value]

   ENVIRONMENT
   -----------
   Algorithm: XGBoost [version]
   Container image: [URI]
   Instance type: [type]
   Region: us-east-1
   IAM role: [ARN]

   MODEL ARTIFACT
   --------------
   S3 URI: [path to model.tar.gz]
   Training job ARN: [ARN]

   METRICS
   -------
   train:auc: [value]
   validation:auc: [value]
   Training duration: [seconds]
   ```

2. Save the report. This document enables any team member to recreate the model by re-running the training job with the same data, parameters, and environment.

---

## Presentation Checkpoint
Be prepared to show:
- The completed reproducibility report with all fields populated.
- How each field was sourced (Experiments UI, Lineage Graph, training job details, or S3).
- An explanation of why each field matters for reproducibility.

---

## Key Concepts
- **Reproducibility:** The ability to recreate an identical model by using the same data, code, parameters, and environment. Critical for audit compliance and debugging.
- **Data Versioning:** Recording the exact S3 path and timestamp (or version ID) of training data ensures the same data can be retrieved even if the bucket contents change.
- **Container Image URI:** The Docker image that ran the training algorithm. Pinning the exact URI (including the tag or digest) prevents silent algorithm updates from changing results.
- **Artifact Provenance:** The combination of data version, hyperparameters, environment, and output location that fully defines how a model was produced.
