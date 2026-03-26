# Guide 6: Cleanup

Delete all inference resources created during this lab to avoid ongoing charges. SageMaker endpoints incur costs every minute they are running, so this guide is mandatory.

---

## Steps

### Step 1 -- Delete All Endpoints

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar under **Inference**, select **Endpoints**.
3. Select `ASM-FraudShield-Serverless-EP` and click **Actions** then **Delete**. Confirm the deletion.
4. Select `ASM-FraudShield-Async-EP` and click **Actions** then **Delete**. Confirm the deletion.
5. Select `ASM-FraudShield-MME-EP` and click **Actions** then **Delete**. Confirm the deletion.
6. Select `ASM-FraudShield-Pipeline-EP` and click **Actions** then **Delete**. Confirm the deletion.
7. Wait until all four endpoints disappear from the list or show "Deleting" status.

---

### Step 2 -- Delete All Endpoint Configurations

1. In the left sidebar under **Inference**, select **Endpoint configurations**.
2. Select and delete each of the following configurations:
   - `ASM-FraudShield-Serverless-EPC`
   - `ASM-FraudShield-Async-EPC`
   - `ASM-FraudShield-MME-EPC`
   - `ASM-FraudShield-Pipeline-EPC`
3. For each, click **Actions** then **Delete** and confirm.

---

### Step 3 -- Delete All Models

1. In the left sidebar under **Inference**, select **Models**.
2. Select and delete each of the following models:
   - `ASM-FraudShield-Serverless-Model`
   - `ASM-FraudShield-MultiModel`
   - `ASM-FraudShield-Preprocessor`
   - `ASM-FraudShield-Pipeline`
3. For each, click **Actions** then **Delete** and confirm.

---

### Step 4 -- Delete S3 Artifacts

1. Navigate to **Amazon S3** in the console.
2. Open your bucket `sagemaker-fraudshield-<account-id>`.
3. Delete the following folders and their contents:
   - `async-input/`
   - `async-output/`
   - `async-failures/`
   - `batch-input/`
   - `batch-output/`
   - `multi-model/`
4. For each folder, select it, click **Delete**, type the confirmation text, and confirm.

---

### Step 5 -- Delete the SNS Topic

1. Navigate to **Amazon SNS** in the console.
2. In the left sidebar, select **Topics**.
3. Select `ASM-FraudShield-AsyncInference-Notifications`.
4. Click **Delete** and confirm.
5. Go to **Subscriptions** in the left sidebar and delete any orphaned subscriptions related to this topic.

---

### Step 6 -- Verify Cleanup

1. Return to **SageMaker** and confirm the Endpoints list shows none of the lab endpoints.
2. Confirm the Endpoint configurations list is clear of lab resources.
3. Confirm the Models list no longer contains lab models.
4. Check S3 to verify the deleted folders are gone.
5. Check SNS to verify the topic has been removed.

---

## Presentation Checkpoint
Be prepared to show:
- The empty (or lab-free) Endpoints list in SageMaker
- The S3 bucket without the lab-specific folders
- The SNS Topics list without the async notification topic

---

## Key Concepts
- **Endpoint Deletion Order:** Always delete the endpoint first, then the endpoint configuration, then the model. Deleting in reverse order will fail because of dependencies.
- **S3 Costs:** S3 storage is inexpensive but output files from batch and async jobs accumulate over time. Clean up to maintain a tidy environment.
- **SNS Cleanup:** Deleting a topic does not automatically delete its subscriptions. Always check for orphaned subscriptions.
