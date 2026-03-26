# Guide 6: Cleanup

Delete all monitoring resources, schedules, baselines, alarms, endpoints, and SNS subscriptions created during this lab. Monitoring schedules launch compute jobs on a recurring basis and will incur charges until deleted.

---

## Steps

### Step 1 -- Delete Monitoring Schedules

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar under **Inference**, select **Model Monitor**.
3. Under the **Data quality** tab, locate `ASM-FraudShield-DQ-Schedule`.
4. Select it and click **Delete**. Confirm the deletion.
5. Under the **Model quality** tab, locate `ASM-FraudShield-MQ-Schedule`.
6. Select it and click **Delete**. Confirm the deletion.
7. Verify both schedules have been removed from the list.

---

### Step 2 -- Delete Baseline Jobs

1. Still in the **Model Monitor** section, navigate to baseline jobs.
2. Locate `ASM-FraudShield-DQ-Baseline` and `ASM-FraudShield-MQ-Baseline`.
3. Note that completed baseline jobs cannot be deleted from the console (they are records), but their output in S3 should be cleaned up in Step 5.

---

### Step 3 -- Delete CloudWatch Alarms

1. Navigate to **Amazon CloudWatch** in the console.
2. In the left sidebar, select **Alarms** then **All alarms**.
3. Select `ASM-FraudShield-DataQuality-Alarm`.
4. Click **Actions** then **Delete**. Confirm the deletion.
5. Verify the alarm no longer appears in the list.

---

### Step 4 -- Delete Endpoints and Configurations

1. Navigate back to **Amazon SageMaker**.
2. Under **Inference**, go to **Endpoints**.
3. Select and delete the FraudShield endpoint used for monitoring. Click **Actions** then **Delete** and confirm.
4. Go to **Endpoint configurations** and delete `ASM-FraudShield-DataCapture-EPC`.
5. Go to **Models** and delete any models created specifically for this lab.

---

### Step 5 -- Delete S3 Artifacts

1. Navigate to **Amazon S3** and open your bucket `sagemaker-fraudshield-<account-id>`.
2. Delete the following folders and their contents:
   - `data-capture/`
   - `baseline-data/`
   - `baseline-output/`
   - `monitoring-output/`
   - `ground-truth/`
3. For each folder, select it, click **Delete**, type the confirmation text, and confirm.

---

### Step 6 -- Delete SNS Topics and Subscriptions

1. Navigate to **Amazon SNS** in the console.
2. In the left sidebar, select **Topics**.
3. Select `ASM-FraudShield-MonitorAlerts` and click **Delete**. Confirm.
4. Go to **Subscriptions** in the left sidebar.
5. Delete any subscriptions that were associated with the deleted topic (they may show as "PendingDeletion" or remain as orphans).
6. Verify the topics and subscriptions lists are clear of lab resources.

---

### Step 7 -- Verify Cleanup

1. Return to **SageMaker Model Monitor** and confirm no active schedules remain.
2. Check **CloudWatch Alarms** and confirm the lab alarm is gone.
3. Check **SageMaker Endpoints** and confirm no lab endpoints are running.
4. Check **S3** and confirm the monitoring-related folders have been removed.
5. Check **SNS** and confirm the topic and subscriptions are deleted.

---

## Presentation Checkpoint
Be prepared to show:
- The Model Monitor page with no active schedules
- The CloudWatch Alarms page with no lab alarms
- The S3 bucket without monitoring-related folders
- The SNS Topics list without lab topics

---

## Key Concepts
- **Schedule Deletion Priority:** Always delete monitoring schedules before deleting the endpoints they monitor. An active schedule targeting a deleted endpoint will produce errors on every execution.
- **Baseline Records:** Baseline job records persist in SageMaker for audit purposes, but the output files in S3 should be cleaned up to avoid storage costs.
- **Orphaned Subscriptions:** Deleting an SNS topic does not always remove its subscriptions immediately. Always check the subscriptions list after deleting a topic.
