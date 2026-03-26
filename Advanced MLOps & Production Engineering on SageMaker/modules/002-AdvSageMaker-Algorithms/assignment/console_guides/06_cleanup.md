# Guide 6: Cleanup

This guide ensures you delete every resource created during this lab to avoid ongoing charges. Follow each step in order.

---

## Steps

### Step 1 -- Stop the HPO Tuning Job (If Still Running)
1. In the SageMaker console, expand **Training** and click **Hyperparameter tuning jobs**.
2. If `fraudshield-xgb-hpo` shows a status of **InProgress**, select it and click **Stop**.
3. Wait for the status to change to **Stopped** or **Completed**.

### Step 2 -- Delete Training Job Models
1. In the left navigation, expand **Inference** and click **Models**.
2. Look for any models automatically created from training jobs (e.g., names starting with `fraudshield-xgboost`, `fraudshield-kmeans`, or `fraudshield-rcf`).
3. Select each model and click **Delete**. Confirm each deletion.
4. If no models were created (because you did not deploy), proceed to the next step.

### Step 3 -- Clean Up S3 Artifacts
1. Open the S3 console in a new tab.
2. Navigate to `s3://fraudshield-data-<account-id>/models/`.
3. Delete the following prefixes and their contents:
   - `models/xgboost/`
   - `models/kmeans/`
   - `models/rcf/`
   - `models/hpo/`
4. Each prefix contains `model.tar.gz` files and potentially output data from the training jobs.
5. Optionally delete the processed data folders if they were created solely for this lab.

### Step 4 -- Verify Training Jobs Are Complete
1. Return to the SageMaker console and click **Training jobs**.
2. Confirm that all jobs from this lab (`fraudshield-xgboost-baseline`, `fraudshield-kmeans-segments`, `fraudshield-rcf-anomaly`, and the 10 HPO child jobs) show **Completed** or **Stopped** status.
3. Training jobs cannot be deleted from the console, but they incur no charges once completed. They will remain as historical records.

### Step 5 -- Delete the HPO Tuning Job (Optional)
1. Navigate back to **Hyperparameter tuning jobs**.
2. Note that HPO tuning jobs cannot be individually deleted from the console. They remain as read-only records.
3. The associated training jobs and models have already been addressed in the previous steps. No ongoing charges accrue from completed HPO jobs.

### Step 6 -- Final Verification
1. Confirm no active endpoints exist under **Inference > Endpoints** that were created from lab models.
2. Confirm S3 model artifacts have been removed.
3. Confirm no running notebook instances or Studio apps remain from this module's work.

---

## Presentation Checkpoint
Be prepared to show:
- The **Models** page confirming no lab models remain.
- The S3 bucket confirming model artifact prefixes have been deleted.
- The **Training jobs** page showing all lab jobs in a terminal state (Completed or Stopped).

---

## Key Concepts
- **Training Job Lifecycle:** Completed training jobs are read-only records that incur no ongoing charges. They cannot be deleted but serve as an audit trail.
- **Model Artifacts:** The `model.tar.gz` files stored in S3 persist after training jobs complete. These must be manually deleted to avoid S3 storage charges.
- **HPO Child Jobs:** Each trial in an HPO job creates a separate training job. All child jobs must be accounted for during cleanup.
- **No Orphaned Endpoints:** The most common source of unexpected charges is a forgotten real-time endpoint. Always verify no endpoints remain.
