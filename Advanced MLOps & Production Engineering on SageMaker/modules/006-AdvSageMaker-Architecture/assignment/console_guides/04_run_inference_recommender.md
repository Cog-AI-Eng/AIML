# Guide 4: Run Inference Recommender

Run an Inference Recommender Default job from the SageMaker console to benchmark your FraudShield model across multiple instance types. Compare the results by latency, throughput, and cost to identify the optimal instance for production deployment.

---

## Steps

### Step 1 -- Register the Model in the Model Registry

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar, expand **SageMaker resources** and select **Model registry**.
3. Click **Create model group**.
4. For **Model group name**, enter `ASM-FraudShield-ModelGroup`.
5. Click **Create**.
6. Click on the model group name, then click **Register model version**.
7. Under **Model artifact location**, enter the S3 path to your FraudShield model:
   `s3://sagemaker-fraudshield-<account-id>/models/fraud-model/output/model.tar.gz`
8. Under **Inference image URI**, enter the XGBoost image:
   `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1`
9. For **Framework**, select `XGBOOST`. For **Framework version**, enter `1.5-1`.
10. Click **Register model**. Approve the model version by setting its status to **Approved**.

---

### Step 2 -- Create a Sample Payload

1. Navigate to **Amazon S3** and open your bucket.
2. Create a folder named `inference-recommender/`.
3. Upload a sample payload file named `sample_payload.csv` containing a representative transaction:
   ```
   50.0,1,0,1,234.56,2,0.85,1200,3
   ```
4. Note the S3 URI: `s3://sagemaker-fraudshield-<account-id>/inference-recommender/sample_payload.csv`.

---

### Step 3 -- Launch an Inference Recommender Job

1. Return to **Amazon SageMaker** and navigate to the **Inference** section.
2. Select **Inference recommender** (or find it under **Inference** in the left sidebar).
3. Click **Create inference recommender job**.
4. For **Job name**, enter `ASM-FraudShield-InfRecommender`.
5. For **Job type**, select **Default** (this benchmarks across a range of instance types automatically).
6. Under **Model**:
   - Select the model from the model registry: `ASM-FraudShield-ModelGroup`, version 1.
7. Under **Sample payload**:
   - **S3 URI:** `s3://sagemaker-fraudshield-<account-id>/inference-recommender/sample_payload.csv`
   - **Content type:** `text/csv`
8. Under **IAM role**, select your SageMaker execution role.
9. Click **Create job**.

---

### Step 4 -- Monitor the Job

1. The job will appear in the Inference Recommender jobs list with status **InProgress**.
2. Click on the job name to view the detail page.
3. The Default job type benchmarks across multiple instance types and may take 20-45 minutes to complete.
4. Wait for the status to change to **Completed**.

---

### Step 5 -- Analyze the Results

1. On the completed job detail page, scroll to the **Results** section.
2. The results table shows each benchmarked instance type with the following metrics:
   - **Instance type** (e.g., ml.m5.large, ml.m5.xlarge, ml.c5.xlarge)
   - **Invocations per minute** (throughput)
   - **Model latency (ms)** (p50, p90, p99)
   - **Estimated cost per hour**
3. Sort by latency to find the fastest option.
4. Sort by cost to find the cheapest option.
5. Identify the best value instance: the one that meets your latency SLA at the lowest cost.
6. Document your recommendation with the supporting data points.

---

## Presentation Checkpoint
Be prepared to show:
- The registered model in the Model Registry with Approved status
- The completed Inference Recommender job
- The results table comparing instance types by latency, throughput, and cost

---

## Key Concepts
- **Inference Recommender:** A SageMaker feature that automatically benchmarks your model on different instance types, helping you choose the right compute for your latency and cost requirements.
- **Default Job:** Benchmarks across a curated set of instance types. Use the Advanced job type if you want to specify a custom list of instances to test.
- **Model Registry:** A versioned catalog of model artifacts. Inference Recommender requires models to be registered so it can track which version was benchmarked.
- **Latency vs. Throughput vs. Cost:** The three axes of the inference optimization trade-off. Larger instances reduce latency and increase throughput but cost more per hour.
