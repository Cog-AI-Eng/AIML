# Guide 1: Deploy a Serverless Inference Endpoint

Deploy your trained FraudShield model as a Serverless Inference endpoint. Serverless endpoints eliminate idle costs by scaling to zero when there is no traffic and automatically provisioning compute when requests arrive.

---

## Steps

### Step 1 -- Create a SageMaker Model

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar, expand **Inference** and select **Models**.
3. Click **Create model**.
4. For **Model name**, enter `ASM-FraudShield-Serverless-Model`.
5. Under **IAM role**, select your existing SageMaker execution role or create a new one with S3 access.
6. Under **Container definition**, choose **Provide model artifacts and inference image options**.
7. For **Inference image URI**, enter the XGBoost built-in algorithm image for us-east-1:
   `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1`
8. For **Model artifact location**, enter the S3 URI of your trained model:
   `s3://sagemaker-fraudshield-<account-id>/models/fraud-model/output/model.tar.gz`
9. Click **Create model**.
10. Verify the model appears in the Models list with status showing as available.

---

### Step 2 -- Create a Serverless Endpoint Configuration

1. In the left sidebar under **Inference**, select **Endpoint configurations**.
2. Click **Create endpoint configuration**.
3. For **Endpoint configuration name**, enter `ASM-FraudShield-Serverless-EPC`.
4. Under **Type of endpoint**, select **Serverless**.
5. Click **Add model** and select `ASM-FraudShield-Serverless-Model`.
6. Configure the serverless settings:
   - **Memory size (MB):** Select `2048` (2 GB).
   - **Max concurrency:** Enter `5`.
7. Leave provisioned concurrency disabled for this lab.
8. Click **Create endpoint configuration**.

---

### Step 3 -- Deploy the Serverless Endpoint

1. In the left sidebar under **Inference**, select **Endpoints**.
2. Click **Create endpoint**.
3. For **Endpoint name**, enter `ASM-FraudShield-Serverless-EP`.
4. Under **Endpoint configuration**, select **Use an existing endpoint configuration**.
5. Choose `ASM-FraudShield-Serverless-EPC` from the list.
6. Click **Create endpoint**.
7. Wait for the endpoint status to transition from "Creating" to **InService** (this may take 3-5 minutes).

---

### Step 4 -- Invoke the Serverless Endpoint

1. Open **AWS CloudShell** from the top navigation bar (or use the SageMaker notebook terminal).
2. Run the following command to invoke the endpoint with a test transaction:
   ```bash
   aws sagemaker-runtime invoke-endpoint \
     --endpoint-name ASM-FraudShield-Serverless-EP \
     --content-type text/csv \
     --body "50.0,1,0,1,234.56,2,0.85,1200,3" \
     --region us-east-1 \
     output.json
   ```
3. View the prediction result:
   ```bash
   cat output.json
   ```
4. The response should contain a fraud probability score between 0 and 1.
5. Return to the SageMaker console and click on the endpoint name to verify the **Invocation count** metric has incremented.

---

## Presentation Checkpoint
Be prepared to show:
- The serverless endpoint in **InService** status
- The endpoint configuration showing 2048 MB memory and max concurrency of 5
- The output of the invoke-endpoint command with a fraud probability score

---

## Key Concepts
- **Serverless Inference:** A deployment option where AWS manages all underlying compute. Instances scale to zero when idle, so you pay only for the duration of each request.
- **Memory Size:** Determines the compute allocated per request. Options range from 1024 MB to 6144 MB. Larger memory also provides more vCPUs.
- **Max Concurrency:** The maximum number of concurrent invocations the endpoint handles before throttling. Requests beyond this limit receive a 429 response.
- **Cold Start:** The first request after a period of inactivity takes longer because AWS must provision compute. Subsequent requests within the keep-alive window are faster.
