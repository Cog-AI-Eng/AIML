# Guide 5: Cleanup

This guide ensures you delete every resource created during this lab to avoid ongoing charges. Follow each step in order.

---

## Steps

### Step 1 -- Delete Experiment Runs
1. In SageMaker Studio, navigate to **Experiments**.
2. Click the experiment `fraudshield-fraud-detection`.
3. Select all three runs: `run1-conservative`, `run2-moderate`, `run3-aggressive`.
4. Click **Delete** (or right-click and select **Delete**). Confirm the deletion.
5. Wait for the runs to be removed from the list.

### Step 2 -- Delete the Experiment
1. Return to the Experiments list.
2. Select `fraudshield-fraud-detection`.
3. Click **Delete** and confirm. The experiment and its metadata are permanently removed.
4. Verify the experiment no longer appears in the list.

### Step 3 -- Delete Models
1. In the SageMaker console (not Studio), navigate to **Inference > Models**.
2. Look for any models created from the three training jobs in this lab.
3. Select each lab model and click **Delete**. Confirm each deletion.
4. If no models were registered (because you did not deploy endpoints), proceed to the next step.

### Step 4 -- Clean Up S3 Model Artifacts
1. Open the S3 console.
2. Navigate to `s3://fraudshield-data-<account-id>/models/experiments/`.
3. Delete all contents under the `experiments/` prefix:
   - `run1/output/model.tar.gz`
   - `run2/output/model.tar.gz`
   - `run3/output/model.tar.gz`
4. Delete the `experiments/` folder itself.
5. If you also created data files specifically for this module, delete those as well.

### Step 5 -- Verify Training Jobs Are Terminal
1. In the SageMaker console, navigate to **Training > Training jobs**.
2. Confirm that `fraudshield-exp-run1-conservative`, `fraudshield-exp-run2-moderate`, and `fraudshield-exp-run3-aggressive` all show **Completed** status.
3. Completed training jobs cannot be deleted but incur no ongoing charges.

### Step 6 -- Final Verification
1. In SageMaker Studio, confirm the **Experiments** list shows no lab experiments.
2. In the SageMaker console, confirm **Inference > Models** has no lab models.
3. In S3, confirm the `models/experiments/` prefix is empty or deleted.
4. Verify no running notebook instances or Studio apps remain that were started during this module (shut them down if present).

---

## Presentation Checkpoint
Be prepared to show:
- The Experiments page in Studio with no lab experiments listed.
- The Models page in the SageMaker console with no lab models.
- The S3 bucket confirming the `experiments/` prefix has been removed.

---

## Key Concepts
- **Experiment Deletion:** Deleting an experiment removes its metadata and run associations but does not automatically delete the underlying training jobs or S3 artifacts.
- **Orphaned Artifacts:** Model artifacts in S3 persist independently of experiments and training jobs. Always delete them manually to avoid storage charges.
- **Training Job Records:** Completed training jobs are retained as immutable audit records. They do not incur compute charges but remain visible in the console.
- **Studio App Costs:** Running Studio apps (JupyterServer, KernelGateway) incur per-hour charges. Always shut them down when not in active use.
