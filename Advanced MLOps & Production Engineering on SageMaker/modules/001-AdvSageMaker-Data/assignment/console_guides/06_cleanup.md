# Guide 6: Cleanup

This guide ensures you delete every resource created during this lab to avoid ongoing charges. Follow each step in order, as some resources must be removed before others.

---

## Steps

### Step 1 -- Delete the Canvas App
1. In the SageMaker console, navigate to **Domains** and click `fraudshield-studio-domain`.
2. Click the `fraudshield-analyst` user profile.
3. Under **Apps**, locate the Canvas app (listed as type **Canvas**).
4. Click **Delete app**. Confirm the deletion when prompted.
5. Wait for the app status to show **Deleted** before proceeding.

### Step 2 -- Delete Data Wrangler Resources
1. In the user profile's **Apps** section, locate any running Data Wrangler or KernelGateway apps.
2. Delete each app by clicking **Delete app** and confirming.
3. In S3, navigate to `s3://fraudshield-data-<account-id>/` and delete any Data Wrangler output files under folders like `data-wrangler/` or `export/`.

### Step 3 -- Delete the SageMaker Pipeline
1. In the SageMaker console left navigation, expand **Pipelines** and click **Pipelines**.
2. Select `fraudshield-data-pipeline`.
3. Click **Delete** and confirm. This removes the pipeline definition and all execution history.

### Step 4 -- Delete the Feature Group
1. In the left navigation, expand **Data** and click **Feature Store**.
2. Select `fraudshield-txn-features`.
3. Click **Delete feature group** and confirm.
4. Navigate to S3 and delete the offline store data under `s3://fraudshield-data-<account-id>/feature-store/`.
5. Optionally, navigate to **AWS Glue > Databases** and delete the auto-created table if you want a fully clean state.

### Step 5 -- Delete the Studio Domain
1. Return to **Domains** in the SageMaker console.
2. Click `fraudshield-studio-domain`.
3. Delete all remaining user profiles first: select `fraudshield-analyst` and click **Delete**.
4. Once all user profiles are deleted, click **Delete domain**.
5. Confirm the deletion. The domain and its associated EFS file system will be removed.
6. This may take several minutes. Wait for the domain to disappear from the list.

### Step 6 -- Verify Resource Removal
1. In the SageMaker console, confirm no domains, pipelines, or feature groups remain from this lab.
2. In S3, confirm the `fraudshield-data-<account-id>` bucket no longer contains lab artifacts (or delete the bucket entirely if it was created solely for this lab).
3. In the IAM console, optionally delete the `AmazonSageMaker-ExecutionRole-...` role if it was created only for this lab.
4. In the VPC console, verify no orphaned security groups or ENIs remain from the Studio domain.

---

## Presentation Checkpoint
Be prepared to show:
- The SageMaker Domains page with no lab domains listed.
- The Feature Store page with no lab feature groups.
- The Pipelines page with no lab pipelines.
- The S3 bucket confirming all lab artifacts have been removed.

---

## Key Concepts
- **Resource Dependencies:** Studio apps must be deleted before user profiles, and user profiles must be deleted before the domain.
- **EFS Cleanup:** Deleting a Studio domain also deletes its associated EFS file system and all user home directories.
- **Offline Store Artifacts:** Feature Store offline data persists in S3 even after the feature group is deleted; you must remove it manually.
- **Cost Avoidance:** Running Studio apps, Feature Store online stores, and idle notebooks all incur charges. Always clean up after labs.
