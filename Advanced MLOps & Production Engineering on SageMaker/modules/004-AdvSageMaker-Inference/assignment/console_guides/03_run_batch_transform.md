# Guide 3: Run a Batch Transform Job

Configure and run a Batch Transform job to score an entire CSV file of historical FraudShield transactions stored in S3. Batch Transform is the right choice when you need offline predictions on large datasets without maintaining a persistent endpoint.

---

## Steps

### Step 1 -- Upload the Historical Transactions Dataset

1. Open the **AWS Management Console** and navigate to **Amazon S3**.
2. Open your bucket `sagemaker-fraudshield-<account-id>`.
3. Create a folder named `batch-input`.
4. Upload your historical transactions CSV file (without headers, features only) to this folder. Name it `historical_transactions.csv`.
5. Confirm the file is visible at `s3://sagemaker-fraudshield-<account-id>/batch-input/historical_transactions.csv`.

---

### Step 2 -- Create a Batch Transform Job

1. Navigate to **Amazon SageMaker** in the console.
2. In the left sidebar, expand **Inference** and select **Batch transform jobs**.
3. Click **Create batch transform job**.
4. For **Job name**, enter `ASM-FraudShield-BatchTransform`.
5. Under **Model name**, select `ASM-FraudShield-Serverless-Model` (or the model created in Guide 1).
6. For **Instance type**, select `ml.m5.xlarge`.
7. For **Instance count**, enter `1`.

---

### Step 3 -- Configure Input and Output Settings

1. Under **Input data configuration**:
   - **S3 data type:** Select `S3Prefix`.
   - **S3 URI:** Enter `s3://sagemaker-fraudshield-<account-id>/batch-input/`.
   - **Content type:** Enter `text/csv`.
   - **Split type:** Select `Line` (each line is a separate record).
   - **Compression type:** Select `None`.
2. Under **Output data configuration**:
   - **S3 output path:** Enter `s3://sagemaker-fraudshield-<account-id>/batch-output/`.
   - **Accept:** Enter `text/csv`.
   - **Assemble with:** Select `Line`.
3. Under **Additional configuration** (optional):
   - **Max payload size (MB):** Leave at default (6 MB).
   - **Max concurrent transforms:** Leave at default (1).
   - **Join source:** Select `Input` if you want each output line paired with its input record.

---

### Step 4 -- Launch and Monitor the Job

1. Click **Create job**.
2. You will be redirected to the Batch transform jobs list. The job status will show **InProgress**.
3. Click on the job name `ASM-FraudShield-BatchTransform` to view details.
4. Monitor the job status. It typically takes 5-10 minutes depending on dataset size.
5. Wait until the status changes to **Completed**.

---

### Step 5 -- Examine the Results

1. Navigate to **Amazon S3** and open your bucket.
2. Open the `batch-output/` folder.
3. Locate the output file (it will be named `historical_transactions.csv.out`).
4. Download or preview the file. Each line contains a fraud probability score corresponding to the matching input row.
5. Verify the number of output lines matches the number of input records.

---

## Presentation Checkpoint
Be prepared to show:
- The Batch Transform job in **Completed** status
- The job detail page showing input S3 path, output S3 path, and instance configuration
- The output file in S3 with prediction scores

---

## Key Concepts
- **Batch Transform:** A managed, ephemeral compute job that reads data from S3, runs inference, writes results to S3, and then terminates. No persistent endpoint is needed.
- **Split Type:** Controls how SageMaker partitions the input. `Line` sends each line as a separate inference request, which is standard for CSV data.
- **Join Source:** When set to `Input`, each output line is prepended with the corresponding input record, making it easy to map predictions back to source data.
- **Max Concurrent Transforms:** Controls parallelism within the job. Increasing this value and instance count can speed up large jobs at additional cost.
