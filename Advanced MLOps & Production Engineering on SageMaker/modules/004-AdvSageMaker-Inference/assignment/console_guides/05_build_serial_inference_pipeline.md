# Guide 5: Build a Serial Inference Pipeline

Create a 2-container serial inference pipeline that chains a preprocessing container with a prediction container. The preprocessor normalizes and transforms raw transaction data before passing it to the FraudShield XGBoost model for scoring, enabling end-to-end inference without client-side preprocessing.

---

## Steps

### Step 1 -- Prepare the Preprocessor Model Artifact

1. Open **Amazon SageMaker** in the console and navigate to **Notebook instances** under **Notebook**.
2. Open your existing notebook instance (or create one using `ml.t3.medium`).
3. In JupyterLab, create a new terminal and build a simple preprocessor artifact:
   ```bash
   mkdir -p preprocessor && cd preprocessor
   ```
4. Create a preprocessing script. For a Scikit-learn pipeline, create the standard `inference.py` that applies a StandardScaler saved in the artifact.
5. Package the preprocessor as `preprocessor-model.tar.gz`:
   ```bash
   tar -czf preprocessor-model.tar.gz -C preprocessor .
   ```
6. Upload to S3:
   ```bash
   aws s3 cp preprocessor-model.tar.gz \
     s3://sagemaker-fraudshield-<account-id>/models/preprocessor/preprocessor-model.tar.gz
   ```

---

### Step 2 -- Create the Preprocessor Model in SageMaker

1. In the SageMaker console, go to **Models** under **Inference** and click **Create model**.
2. For **Model name**, enter `ASM-FraudShield-Preprocessor`.
3. Under **Container definition**:
   - **Inference image URI:** Use the Scikit-learn image:
     `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3`
   - **Model artifact location:** `s3://sagemaker-fraudshield-<account-id>/models/preprocessor/preprocessor-model.tar.gz`
4. Click **Create model**.

---

### Step 3 -- Create a Pipeline Model (Inference Pipeline)

1. Navigate to **Models** and click **Create model**.
2. For **Model name**, enter `ASM-FraudShield-Pipeline`.
3. Under **Container definition**, select **Add container** to add the first container:
   - **Container 1 (Preprocessor):**
     - **Inference image URI:** Scikit-learn image (same as Step 2).
     - **Model artifact location:** `s3://sagemaker-fraudshield-<account-id>/models/preprocessor/preprocessor-model.tar.gz`
4. Click **Add container** to add the second container:
   - **Container 2 (Predictor):**
     - **Inference image URI:** XGBoost image:
       `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1`
     - **Model artifact location:** `s3://sagemaker-fraudshield-<account-id>/models/fraud-model/output/model.tar.gz`
5. Verify containers appear in order: Preprocessor first, Predictor second.
6. Click **Create model**.

---

### Step 4 -- Deploy the Pipeline Endpoint

1. Go to **Endpoint configurations** and click **Create endpoint configuration**.
2. For **Endpoint configuration name**, enter `ASM-FraudShield-Pipeline-EPC`.
3. Select **Real-time** endpoint type.
4. Click **Add model** and select `ASM-FraudShield-Pipeline`.
5. For **Instance type**, select `ml.m5.xlarge`. Set **Initial instance count** to `1`.
6. Click **Create endpoint configuration**.
7. Go to **Endpoints**, click **Create endpoint**.
8. For **Endpoint name**, enter `ASM-FraudShield-Pipeline-EP`.
9. Select `ASM-FraudShield-Pipeline-EPC` and click **Create endpoint**.
10. Wait for **InService** status (5-10 minutes).

---

### Step 5 -- Test End-to-End Invocation

1. Open **AWS CloudShell** and invoke the pipeline endpoint with raw (unprocessed) transaction data:
   ```bash
   aws sagemaker-runtime invoke-endpoint \
     --endpoint-name ASM-FraudShield-Pipeline-EP \
     --content-type text/csv \
     --body "50.0,1,0,1,234.56,2,0.85,1200,3" \
     --region us-east-1 \
     pipeline-output.json
   ```
2. View the result:
   ```bash
   cat pipeline-output.json
   ```
3. The data flows through the preprocessor container (normalization/transformation) and then through the XGBoost container (prediction), returning a fraud probability score.
4. Verify the response contains a valid prediction value.

---

## Presentation Checkpoint
Be prepared to show:
- The pipeline model detail page showing two containers in serial order
- The pipeline endpoint in **InService** status
- The output of the invoke-endpoint call proving end-to-end data flow

---

## Key Concepts
- **Inference Pipeline:** A SageMaker model composed of 2-15 containers arranged in a linear sequence. Each container's output becomes the next container's input.
- **Preprocessing Container:** Handles feature engineering, normalization, or data transformation so the client sends raw data and the pipeline handles all transformations server-side.
- **Container Chaining:** SageMaker manages the inter-container data transfer automatically. Each container exposes the standard `/invocations` endpoint and receives the previous container's output as its input.
- **Single Endpoint:** The entire pipeline deploys behind one endpoint, simplifying client integration and ensuring preprocessing logic stays in sync with the model.
