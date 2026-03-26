# Guide 1: Train XGBoost in Algorithm Mode

This guide walks you through launching an XGBoost training job directly from the SageMaker console using Algorithm Mode. You will configure hyperparameters, specify S3 input and output channels, and monitor the job to completion -- all without writing a training script.

---

## Steps

### Step 1 -- Navigate to Training Jobs
1. Sign in to the AWS Management Console and set your region to **us-east-1**.
2. Open the **Amazon SageMaker** service.
3. In the left navigation pane, expand **Training** and click **Training jobs**.
4. Click **Create training job**.

### Step 2 -- Configure Job Settings
1. Enter the training job name: `fraudshield-xgboost-baseline`.
2. Under **IAM role**, select the SageMaker execution role from Module 1.
3. Under **Algorithm source**, select **SageMaker built-in algorithm**.
4. From the algorithm list, select **XGBoost** (look for the latest version, e.g., 1.5-1 or newer).
5. Under **Input mode**, select **File**.

### Step 3 -- Set Hyperparameters
1. Scroll to the **Hyperparameters** section and configure the following:
   - `objective`: `binary:logistic`
   - `num_round`: `100`
   - `max_depth`: `5`
   - `eta`: `0.2`
   - `subsample`: `0.8`
   - `colsample_bytree`: `0.8`
   - `eval_metric`: `auc`
2. Leave all other hyperparameters at their default values.

### Step 4 -- Configure Resource Limits
1. Under **Resource configuration**, set the instance type to `ml.m5.xlarge`.
2. Set the instance count to `1`.
3. Set the volume size to `10` GB.
4. Under **Stopping condition**, set the maximum runtime to `3600` seconds (1 hour).

### Step 5 -- Specify Input Data Channels
1. Scroll to **Input data configuration** and click **Add channel**.
2. For the **train** channel:
   - Channel name: `train`
   - Data source: **S3**
   - S3 data type: **S3Prefix**
   - S3 URI: `s3://fraudshield-data-<account-id>/processed/train/`
   - Content type: `text/csv`
3. Click **Add channel** again for the **validation** channel:
   - Channel name: `validation`
   - S3 URI: `s3://fraudshield-data-<account-id>/processed/validation/`
   - Content type: `text/csv`

### Step 6 -- Specify Output Location
1. Under **Output data configuration**, set the S3 output path to `s3://fraudshield-data-<account-id>/models/xgboost/`.
2. Leave encryption settings at defaults.

### Step 7 -- Launch and Monitor
1. Click **Create training job**.
2. The job status changes to **InProgress**. Click the job name to view details.
3. Scroll to the **Monitor** section to watch CloudWatch metrics (train:auc, validation:auc) update in near real-time.
4. Wait for the job to reach **Completed** status (typically 3-8 minutes with this dataset size).
5. Under **Output**, note the S3 path to the `model.tar.gz` artifact.

---

## Presentation Checkpoint
Be prepared to show:
- The completed training job detail page with status **Completed**.
- The hyperparameter values configured for the job.
- The training and validation AUC metrics from the CloudWatch charts.

---

## Key Concepts
- **Algorithm Mode:** A SageMaker training mode where you select a built-in algorithm and configure it entirely through parameters -- no custom training script required.
- **XGBoost:** A gradient-boosted tree algorithm widely used for tabular classification and regression. SageMaker provides a managed, distributed implementation.
- **Input Channels:** Named S3 paths that supply training and validation data to the training container. XGBoost expects `train` and optionally `validation` channels.
- **eval_metric (AUC):** Area Under the ROC Curve, a metric that measures how well the model separates positive (fraud) and negative (legitimate) classes.
