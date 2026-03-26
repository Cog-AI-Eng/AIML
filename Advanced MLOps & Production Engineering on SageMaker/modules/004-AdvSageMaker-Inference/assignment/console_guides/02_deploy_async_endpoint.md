# Guide 2: Deploy an Async Inference Endpoint

Create an Asynchronous Inference endpoint for FraudShield workloads where predictions can tolerate seconds to minutes of latency. Async endpoints are ideal for large payload scoring and cost-sensitive batch-like workloads that still need near-real-time results.

---

## Steps

### Step 1 -- Create an SNS Topic for Notifications

1. Open the **AWS Management Console** and navigate to **Amazon SNS**.
2. In the left sidebar, select **Topics**, then click **Create topic**.
3. Select **Standard** as the topic type.
4. For **Name**, enter `ASM-FraudShield-AsyncInference-Notifications`.
5. Click **Create topic**.
6. On the topic detail page, click **Create subscription**.
7. For **Protocol**, select **Email**. For **Endpoint**, enter your email address.
8. Click **Create subscription**, then check your inbox and confirm the subscription.
9. Copy the **Topic ARN** -- you will need it in the next step.

---

### Step 2 -- Create an Async Endpoint Configuration

1. Navigate to **Amazon SageMaker** and expand **Inference** in the left sidebar.
2. Select **Endpoint configurations** and click **Create endpoint configuration**.
3. For **Endpoint configuration name**, enter `ASM-FraudShield-Async-EPC`.
4. Under **Type of endpoint**, select **Asynchronous**.
5. Click **Add model** and select `ASM-FraudShield-Serverless-Model` (created in Guide 1) or create a new model if needed.
6. For **Instance type**, select `ml.m5.xlarge`. Set **Initial instance count** to `1`.
7. Under **Async inference output configuration**:
   - **S3 output path:** `s3://sagemaker-fraudshield-<account-id>/async-output/`
   - **S3 failure path:** `s3://sagemaker-fraudshield-<account-id>/async-failures/`
8. Under **Notification configuration (optional)**:
   - **Success SNS topic ARN:** Paste the ARN from Step 1.
   - **Error SNS topic ARN:** Paste the same ARN.
9. Click **Create endpoint configuration**.

---

### Step 3 -- Deploy the Async Endpoint

1. In the left sidebar under **Inference**, select **Endpoints**.
2. Click **Create endpoint**.
3. For **Endpoint name**, enter `ASM-FraudShield-Async-EP`.
4. Select **Use an existing endpoint configuration** and choose `ASM-FraudShield-Async-EPC`.
5. Click **Create endpoint**.
6. Wait for the endpoint status to reach **InService** (approximately 5-8 minutes).

---

### Step 4 -- Upload a Test Payload to S3

1. Navigate to **Amazon S3** in the console.
2. Open your bucket `sagemaker-fraudshield-<account-id>`.
3. Create a folder named `async-input`.
4. Upload a CSV file named `test_transaction.csv` containing a single transaction row:
   ```
   50.0,1,0,1,234.56,2,0.85,1200,3
   ```
5. Note the full S3 URI: `s3://sagemaker-fraudshield-<account-id>/async-input/test_transaction.csv`.

---

### Step 5 -- Submit an Async Invocation and Retrieve Results

1. Open **AWS CloudShell** and run the following command:
   ```bash
   aws sagemaker-runtime invoke-endpoint-async \
     --endpoint-name ASM-FraudShield-Async-EP \
     --input-location s3://sagemaker-fraudshield-<account-id>/async-input/test_transaction.csv \
     --content-type text/csv \
     --region us-east-1
   ```
2. The response returns an `OutputLocation` S3 URI. Copy this URI.
3. Wait 30-60 seconds, then check the output in S3:
   ```bash
   aws s3 cp <OutputLocation-URI> ./async_result.json
   cat async_result.json
   ```
4. Verify you received a fraud probability score.
5. Check your email for the SNS success notification.

---

## Presentation Checkpoint
Be prepared to show:
- The Async endpoint in **InService** status
- The SNS topic with a confirmed email subscription
- The output file in S3 containing the prediction result
- The SNS notification email confirming successful inference

---

## Key Concepts
- **Async Inference:** Designed for requests with large payloads or long processing times. The client submits input via S3 and polls for results rather than waiting synchronously.
- **S3 Output Location:** Where SageMaker writes prediction results. Each invocation produces a uniquely named output file.
- **SNS Notification:** Optional integration that sends a message when an async invocation succeeds or fails, eliminating the need to poll.
- **Scale-to-Zero:** Async endpoints can scale down to zero instances during idle periods when configured with auto-scaling, further reducing cost.
