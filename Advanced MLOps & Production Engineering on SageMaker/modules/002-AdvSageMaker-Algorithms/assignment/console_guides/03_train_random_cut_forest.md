# Guide 3: Train Random Cut Forest

This guide walks you through training a Random Cut Forest (RCF) anomaly detection model on FraudShield time-series transaction data. RCF assigns an anomaly score to each data point, enabling the team to flag unusual transactions that deviate from normal spending patterns.

---

## Steps

### Step 1 -- Prepare Time-Series Data
1. Ensure your time-series transaction file is uploaded to S3. The file should contain a single numeric column (e.g., `transaction_amount` aggregated by hour) or multiple numeric feature columns.
2. The data should be in CSV format with no header row.
3. Verify the file is at `s3://fraudshield-data-<account-id>/processed/anomaly/timeseries_transactions.csv`.

### Step 2 -- Create the Training Job
1. In the SageMaker console, navigate to **Training > Training jobs**.
2. Click **Create training job**.
3. Enter the job name: `fraudshield-rcf-anomaly`.
4. Under **IAM role**, select the SageMaker execution role.
5. Under **Algorithm source**, select **SageMaker built-in algorithm**.
6. From the list, select **Random Cut Forest**.

### Step 3 -- Set Hyperparameters
1. In the **Hyperparameters** section, configure:
   - `feature_dim`: enter the number of numeric columns in your data (e.g., `1` for univariate or `4` for multivariate)
   - `num_trees`: `100` (more trees improve accuracy at the cost of training time)
   - `num_samples_per_tree`: `256`
   - `eval_metrics`: `accuracy` (optional, enables evaluation logging)
2. Leave other parameters at their defaults.

### Step 4 -- Configure Resources and Input
1. Under **Resource configuration**, set instance type to `ml.m5.xlarge`, instance count to `1`, volume size to `10` GB.
2. Under **Stopping condition**, set maximum runtime to `3600` seconds.
3. Under **Input data configuration**, add a **train** channel:
   - S3 URI: `s3://fraudshield-data-<account-id>/processed/anomaly/`
   - Content type: `text/csv`
   - S3 data type: **S3Prefix**
4. Under **Output data configuration**, set the output path to `s3://fraudshield-data-<account-id>/models/rcf/`.

### Step 5 -- Launch and Monitor
1. Click **Create training job**.
2. The job status changes to **InProgress**. RCF training on moderate datasets typically completes in 3-7 minutes.
3. Monitor the training progress in the **Monitor** section. The algorithm does not report a traditional loss curve but logs resource utilization.
4. Wait for the job to show **Completed** status.

### Step 6 -- Understand the Output
1. Once training completes, note the output `model.tar.gz` path in S3.
2. The trained RCF model can be deployed to an endpoint. When invoked with new data, it returns an anomaly score for each record.
3. Higher anomaly scores indicate data points that are more unusual relative to the training distribution. The team can set a threshold to flag scores above a chosen cutoff as potential fraud.
4. Review the training job details page and note the total training time and billable seconds.

---

## Presentation Checkpoint
Be prepared to show:
- The completed RCF training job with status **Completed**.
- The hyperparameters configured, especially `num_trees` and `feature_dim`.
- An explanation of how anomaly scores work and how the team would set a detection threshold.

---

## Key Concepts
- **Random Cut Forest (RCF):** An unsupervised anomaly detection algorithm that isolates outliers by randomly partitioning feature space with binary trees. Points that require fewer cuts to isolate receive higher anomaly scores.
- **Anomaly Score:** A numeric value assigned to each data point. Points far from the learned data distribution receive high scores; points consistent with normal patterns receive low scores.
- **num_trees:** The number of random trees in the forest. More trees improve score stability and accuracy but increase training time and model size.
- **num_samples_per_tree:** The number of data points sampled to build each tree. Controls the trade-off between model granularity and overfitting to noise.
