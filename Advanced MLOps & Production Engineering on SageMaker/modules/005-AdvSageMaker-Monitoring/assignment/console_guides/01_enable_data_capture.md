# Guide 1: Enable Data Capture

Enable data capture on an existing FraudShield real-time endpoint so that inference requests and responses are automatically logged to S3. Data capture is the foundation for all Model Monitor functionality.

---

## Steps

### Step 1 -- Identify the Target Endpoint

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar under **Inference**, select **Endpoints**.
3. Locate your active FraudShield endpoint (e.g., one created in a previous module). If no endpoint exists, create a real-time endpoint using `ml.m5.xlarge` with your trained FraudShield model before proceeding.
4. Click on the endpoint name and note its current status is **InService**.
5. Note the endpoint name -- you will reference it throughout this lab.

---

### Step 2 -- Update the Endpoint to Enable Data Capture

1. Since the console does not support editing data capture on an existing endpoint directly, you will create a new endpoint configuration with data capture enabled, then update the endpoint.
2. Go to **Endpoint configurations** and click **Create endpoint configuration**.
3. For **Endpoint configuration name**, enter `ASM-FraudShield-DataCapture-EPC`.
4. Select **Real-time** as the endpoint type.
5. Click **Add model** and select your FraudShield model.
6. For **Instance type**, select `ml.m5.xlarge`. Set **Initial instance count** to `1`.
7. Under **Data capture**, toggle it to **Enabled**.
8. Configure the following settings:
   - **Capture percentage:** `100` (capture all requests during the lab).
   - **S3 capture upload destination:** `s3://sagemaker-fraudshield-<account-id>/data-capture/`
   - **Capture options:** Check both **Input** and **Output**.
   - **CSV content type:** Ensure `text/csv` is listed.
9. Click **Create endpoint configuration**.

---

### Step 3 -- Update the Endpoint with the New Configuration

1. Go to **Endpoints**, select your FraudShield endpoint.
2. Click **Update endpoint**.
3. Select `ASM-FraudShield-DataCapture-EPC` as the new endpoint configuration.
4. Click **Update endpoint**.
5. The endpoint will briefly enter **Updating** status. Wait until it returns to **InService** (3-5 minutes).

---

### Step 4 -- Send Test Invocations to Generate Captured Data

1. Open **AWS CloudShell** and send several test invocations:
   ```bash
   for i in $(seq 1 10); do
     aws sagemaker-runtime invoke-endpoint \
       --endpoint-name <your-endpoint-name> \
       --content-type text/csv \
       --body "50.0,1,0,1,234.56,2,0.85,1200,$i" \
       --region us-east-1 \
       /dev/null
   done
   ```
2. Wait 2-3 minutes for the captured data to be flushed to S3.

---

### Step 5 -- Verify Captured Data in S3

1. Navigate to **Amazon S3** and open your bucket `sagemaker-fraudshield-<account-id>`.
2. Navigate into the `data-capture/` folder.
3. Drill down through the folder hierarchy (organized by endpoint name, variant, year, month, day, hour).
4. Open one of the `.jsonl` files. Each line contains a JSON object with the captured input and output.
5. Verify that both the request payload (input) and the prediction result (output) appear in the captured record.

---

## Presentation Checkpoint
Be prepared to show:
- The endpoint configuration detail page with data capture enabled at 100%
- The S3 folder structure under `data-capture/` with `.jsonl` files
- The contents of a captured record showing both input and output

---

## Key Concepts
- **Data Capture:** A SageMaker feature that logs inference requests and responses to S3 in real time. It is required for Model Monitor to function.
- **Capture Percentage:** The fraction of requests captured. Use 100% for testing and lower values (e.g., 10-30%) in production to balance observability with storage costs.
- **JSONL Format:** Captured data is stored as JSON Lines files, where each line is a self-contained JSON document containing the encoded input and output payloads.
- **S3 Partitioning:** Captured files are organized by endpoint/variant/date/hour, making it easy to query specific time windows.
