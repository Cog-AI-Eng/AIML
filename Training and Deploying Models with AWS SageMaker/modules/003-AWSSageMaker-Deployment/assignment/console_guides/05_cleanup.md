# Guide 5: Clean Up in the Correct Order

This is the most critical guide in the deployment lab. Endpoints incur continuous charges -- approximately $0.23/hour for `ml.m5.xlarge`. You must delete all deployment resources before stepping away. The cleanup order matters because of the reference chain between objects.

**Required cleanup order:**
1. **Endpoint** (first -- this is the live resource consuming compute)
2. **Endpoint Configuration** (references the model)
3. **Model** (references S3 artifacts and container image)

---

## Steps

### Step 1 -- Delete the Endpoint

1. In the **SageMaker console**, go to **Inference** -> **Endpoints**.
2. Find `fraud-rf-v1-endpoint`.
3. Check the status -- it should be **InService**.
4. Select it (checkbox) and click **Actions** -> **Delete**.
5. Confirm the deletion.
6. Wait for the endpoint to disappear from the list (1-2 minutes).
7. Refresh the page to confirm it is gone.

Once the endpoint is deleted, billing for that instance stops immediately.

### Step 2 -- Delete the Endpoint Configuration

1. Go to **Inference** -> **Endpoint configurations**.
2. Find `fraud-rf-v1-config`.
3. Select it and click **Actions** -> **Delete**.
4. Confirm the deletion.

Endpoint configurations do not incur charges, but leaving them creates clutter and confusion. Clean environments are easier to manage.

### Step 3 -- Delete the Model

1. Go to **Inference** -> **Models**.
2. Find `fraud-rf-v1`.
3. Select it and click **Actions** -> **Delete**.
4. Confirm the deletion.

Model objects do not incur charges either, but deleting them ensures no one accidentally deploys a stale model.

### Step 4 -- Verify All Inference Resources Are Gone

Check each section:

| Section | Expected State |
|---------|---------------|
| **Inference -> Endpoints** | No endpoints from this lab |
| **Inference -> Endpoint configurations** | No configs from this lab |
| **Inference -> Models** | No models from this lab |

### Step 5 -- Decide What to Keep in the Model Registry

1. Go to **Governance** -> **Model registry** -> `fraud-detection-rf`.
2. Version 1 should still be listed with **Approved** status.
3. **Do NOT delete the registry entry** -- it does not incur charges and represents your model's version history. You would use this registry in a real workflow to track all versions of the fraud detection model.

### Step 6 -- Decide What to Keep in S3

1. Navigate to **S3** and open your `fraudshield-training-data-<initials>` bucket.
2. The `data/` prefix contains your training data and the `output/` prefix contains your model artifact.
3. **Keep these for now** -- S3 storage costs are minimal (fractions of a penny per GB/month) and these artifacts demonstrate your training lineage.
4. In a real project, you would implement lifecycle policies to automatically archive or delete old artifacts.

### Step 7 -- Verify Billing

1. Navigate to **AWS Billing and Cost Management** (search for "Billing").
2. Click **Bills** in the left navigation.
3. Look for SageMaker charges. Your costs should reflect only the time the endpoint was running (typically a few cents for this lab).
4. Note: Charges may take a few hours to appear.

### Step 8 -- Keep the Studio Domain

Your Studio Domain, user profiles, and IAM roles should remain in place for Module 4. These do not incur charges when idle.

---

## What to Keep vs. What to Delete

| Resource | Keep? | Reason |
|----------|-------|--------|
| `fraud-rf-v1-endpoint` | **Delete** | Active compute charges |
| `fraud-rf-v1-config` | **Delete** | No charges, but clutter |
| `fraud-rf-v1` (model) | **Delete** | No charges, but prevents stale deployments |
| Model Registry (`fraud-detection-rf`) | **Keep** | No charges; version history has value |
| S3 artifacts (`model.tar.gz`, training data) | **Keep** | Minimal cost; needed for lineage |
| Studio Domain + profiles | **Keep** | Needed for Module 4 |
| IAM roles | **Keep** | Needed for Module 4 |

---

## Presentation Checkpoint

Be prepared to show:
- The **Inference -> Endpoints** page with no active endpoints
- The **Inference -> Endpoint configurations** and **Models** pages with no leftover resources
- The **Model Registry** still containing your registered model version (this is intentional -- governance data persists)
- The **Billing** page (or explain that charges have not yet appeared)
- Explain: Why must you delete the Endpoint before the Configuration and Model? (The endpoint holds a reference to the configuration, which holds a reference to the model. Deleting in this order avoids dangling references. More importantly, the endpoint is the only object actively consuming compute and generating charges.)
- Explain: Why do we keep the Model Registry entry but delete the Model object? (The Model object under Inference is a deployment artifact -- it connects a container and an S3 path for serving. The Registry entry is a governance artifact -- it tracks versions, approval status, and metadata. They serve different purposes.)

---

## Key Concepts

- **Cleanup Order:** Endpoint -> Configuration -> Model. Always delete the outermost, most expensive resource first.
- **Charges vs. Metadata:** Endpoints incur charges; Registry entries, S3 artifacts, and IAM roles do not (or incur negligible storage costs). Clean up compute, keep governance data.
- **Defense Against Billing Surprises:** Make it a habit to check **Inference -> Endpoints** before ending any session. If anything shows **InService**, delete it.
