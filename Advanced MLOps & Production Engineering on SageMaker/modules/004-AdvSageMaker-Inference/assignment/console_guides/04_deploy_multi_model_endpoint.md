# Guide 4: Deploy a Multi-Model Endpoint

Deploy a Multi-Model Endpoint (MME) that hosts three regional FraudShield model variants behind a single endpoint. MMEs dynamically load and unload models from S3 on demand, dramatically reducing hosting costs when you have many models with intermittent traffic.

---

## Steps

### Step 1 -- Upload Three Model Artifacts to a Shared S3 Prefix

1. Open the **AWS Management Console** and navigate to **Amazon S3**.
2. Open your bucket `sagemaker-fraudshield-<account-id>`.
3. Create a folder named `multi-model/`.
4. Upload three model artifacts into this folder. If you only have one trained model, copy it three times with different names to simulate regional variants:
   - `multi-model/fraud-model-us-east.tar.gz`
   - `multi-model/fraud-model-us-west.tar.gz`
   - `multi-model/fraud-model-eu-west.tar.gz`
5. Verify all three artifacts appear under the `multi-model/` prefix.

---

### Step 2 -- Create a Multi-Model SageMaker Model

1. Navigate to **Amazon SageMaker** and select **Models** under **Inference**.
2. Click **Create model**.
3. For **Model name**, enter `ASM-FraudShield-MultiModel`.
4. Under **IAM role**, select your SageMaker execution role.
5. Under **Container definition**, choose **Provide model artifacts and inference image options**.
6. For **Inference image URI**, enter the XGBoost image:
   `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1`
7. For **Model data URL**, enter the S3 prefix (not a specific file):
   `s3://sagemaker-fraudshield-<account-id>/multi-model/`
8. Under **Container mode**, select **MultiModel**.
9. Click **Create model**.

---

### Step 3 -- Create Endpoint Configuration and Deploy

1. Go to **Endpoint configurations** and click **Create endpoint configuration**.
2. For **Endpoint configuration name**, enter `ASM-FraudShield-MME-EPC`.
3. Under **Type of endpoint**, select **Real-time**.
4. Click **Add model** and select `ASM-FraudShield-MultiModel`.
5. For **Instance type**, select `ml.m5.xlarge`. Set **Initial instance count** to `1`.
6. Click **Create endpoint configuration**.
7. Go to **Endpoints** and click **Create endpoint**.
8. For **Endpoint name**, enter `ASM-FraudShield-MME-EP`.
9. Select `ASM-FraudShield-MME-EPC` as the endpoint configuration.
10. Click **Create endpoint** and wait for **InService** status (5-8 minutes).

---

### Step 4 -- Invoke with TargetModel Parameter

1. Open **AWS CloudShell**.
2. Invoke the endpoint specifying the US East model:
   ```bash
   aws sagemaker-runtime invoke-endpoint \
     --endpoint-name ASM-FraudShield-MME-EP \
     --content-type text/csv \
     --body "50.0,1,0,1,234.56,2,0.85,1200,3" \
     --target-model "fraud-model-us-east.tar.gz" \
     --region us-east-1 \
     mme-output-east.json
   ```
3. View the result:
   ```bash
   cat mme-output-east.json
   ```
4. Invoke again with a different target model:
   ```bash
   aws sagemaker-runtime invoke-endpoint \
     --endpoint-name ASM-FraudShield-MME-EP \
     --content-type text/csv \
     --body "50.0,1,0,1,234.56,2,0.85,1200,3" \
     --target-model "fraud-model-eu-west.tar.gz" \
     --region us-east-1 \
     mme-output-eu.json
   ```
5. Compare outputs from both invocations to confirm the endpoint dynamically loaded each model.

---

## Presentation Checkpoint
Be prepared to show:
- The three model artifacts in the shared `multi-model/` S3 prefix
- The Multi-Model Endpoint in **InService** status
- Output from two invoke-endpoint calls using different `--target-model` values

---

## Key Concepts
- **Multi-Model Endpoint (MME):** A single endpoint that can serve predictions from hundreds or thousands of models. Models are loaded into memory on demand from a shared S3 prefix.
- **TargetModel:** The parameter passed at invocation time that tells the endpoint which specific model artifact to load and use for the request.
- **Dynamic Model Loading:** The first invocation for a given model incurs a loading delay. Subsequent requests for the same model are served from memory until the model is evicted due to memory pressure.
- **Cost Efficiency:** Instead of deploying a separate endpoint per model, MME shares compute across all models, reducing cost proportionally to the number of models hosted.
