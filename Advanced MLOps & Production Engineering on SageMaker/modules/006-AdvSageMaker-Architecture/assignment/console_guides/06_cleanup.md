# Guide 6: Cleanup

Delete all architecture lab resources including the CloudWatch dashboard, scaling policies, endpoints, training jobs, and S3 artifacts. Leaving endpoints running or scaling policies active will continue to incur charges.

---

## Steps

### Step 1 -- Delete the CloudWatch Dashboard

1. Open the **AWS Management Console** and navigate to **Amazon CloudWatch**.
2. In the left sidebar, select **Dashboards**.
3. Select `ASM-FraudShield-Operations`.
4. Click **Delete** and confirm the deletion.
5. Verify the dashboard no longer appears in the list.

---

### Step 2 -- Delete CloudWatch Alarms

1. In CloudWatch, go to **Alarms** then **All alarms**.
2. Select and delete any alarms created during this lab:
   - `ASM-FraudShield-5xx-Alarm` (if created)
   - Any auto-generated alarms from the scaling policy (these are typically removed automatically when the scaling policy is deleted, but verify).
3. For each alarm, click **Actions** then **Delete** and confirm.

---

### Step 3 -- Remove Auto-Scaling Policies

1. Navigate to **Amazon SageMaker** and go to **Endpoints**.
2. Click on your FraudShield endpoint.
3. Under **Endpoint runtime settings**, select the production variant.
4. Click **Configure auto scaling** (or **Edit auto scaling**).
5. Remove the scaling policy `ASM-FraudShield-ScalingPolicy`.
6. Set both min and max instance count back to `1` (or simply delete the endpoint in the next step).
7. Click **Save**.

---

### Step 4 -- Delete Endpoints and Configurations

1. In SageMaker, go to **Endpoints**.
2. Select and delete any endpoints used during this lab. Click **Actions** then **Delete** and confirm.
3. Go to **Endpoint configurations** and delete any configurations created during this lab.
4. Go to **Models** and delete any models created during this lab.

---

### Step 5 -- Delete the Model Registry Entry

1. In SageMaker, go to **Model registry**.
2. Click on `ASM-FraudShield-ModelGroup`.
3. Select the model version and click **Delete**. Confirm.
4. After all versions are deleted, delete the model group itself.

---

### Step 6 -- Delete S3 Artifacts

1. Navigate to **Amazon S3** and open your bucket `sagemaker-fraudshield-<account-id>`.
2. Delete the following folders and their contents:
   - `checkpoints/`
   - `models/spot-training/`
   - `models/right-sizing/`
   - `inference-recommender/`
3. For each folder, select it, click **Delete**, type the confirmation text, and confirm.
4. If no other labs depend on the bucket, consider deleting the entire bucket.

---

### Step 7 -- Verify Cleanup

1. Check **SageMaker Endpoints** -- confirm no lab endpoints remain.
2. Check **SageMaker Model registry** -- confirm the model group is deleted.
3. Check **CloudWatch Dashboards** -- confirm the lab dashboard is gone.
4. Check **CloudWatch Alarms** -- confirm no lab alarms remain.
5. Check **S3** -- confirm the lab folders have been removed.

---

## Presentation Checkpoint
Be prepared to show:
- The CloudWatch Dashboards list without `ASM-FraudShield-Operations`
- The SageMaker Endpoints list without any lab endpoints
- The S3 bucket without the lab-specific folders
- The Model Registry without the FraudShield model group

---

## Key Concepts
- **Scaling Policy Cleanup:** Deleting an endpoint does not always remove its associated scaling policy from Application Auto Scaling. Always remove the policy explicitly or verify it was cleaned up.
- **Model Registry Cleanup:** Delete model versions before deleting the model group. A group with active versions cannot be deleted.
- **Training Job Records:** Completed training jobs are historical records in SageMaker and cannot be deleted from the console. However, their output artifacts in S3 should be removed to avoid storage costs.
- **Dashboard Cost:** Dashboards incur a monthly charge. Always delete dashboards you no longer need.
