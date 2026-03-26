# Guide 6: Clean Up All Resources

This is the most important guide in the lab. Running endpoints incur ongoing charges. You must delete all resources created during this lab before stepping away.

**Cleanup order matters.** You must delete resources in this sequence because each object may reference the one below it:

1. **Endpoint** (first -- this is the live resource consuming compute)
2. **Endpoint Configuration** (references the model)
3. **Model** (references S3 artifacts and container image)

---

## Steps

### Step 1 -- Delete the Endpoint

1. In the **SageMaker console**, go to **Inference** -> **Endpoints**.
2. Find `fraudshield-demo-endpoint`.
3. Select it (checkbox) and click **Actions** -> **Delete**.
4. Confirm the deletion.
5. Wait for the endpoint to disappear from the list. This may take 1-2 minutes.

### Step 2 -- Delete the Endpoint Configuration

1. Go to **Inference** -> **Endpoint configurations**.
2. Find the configuration that was associated with your endpoint (it will have a name matching or similar to `fraudshield-demo-endpoint`).
3. Select it and click **Actions** -> **Delete**.
4. Confirm the deletion.

### Step 3 -- Delete the Model

1. Go to **Inference** -> **Models**.
2. Find the model that was created by JumpStart for your deployment.
3. Select it and click **Actions** -> **Delete**.
4. Confirm the deletion.

### Step 4 -- Verify All Inference Resources Are Gone

1. Check each section under **Inference**:
   - **Endpoints:** Should show no endpoints with today's creation date
   - **Endpoint configurations:** Should show no configurations from today
   - **Models:** Should show no models from today
2. If any resources remain, delete them now.

### Step 5 -- Verify IAM Roles (Optional Cleanup)

The IAM roles you created will remain but do not incur charges. For a clean environment:

1. Navigate to **IAM** -> **Roles**.
2. The **SageMaker-FraudShield-Training** role you created in Guide 4 can remain -- it will be useful in Module 2.
3. The auto-created **AmazonSageMaker-ExecutionRole** should also remain -- it is needed by the Studio Domain.

### Step 6 -- Check the Studio Domain (Do Not Delete)

1. Go to **SageMaker** -> **Domains**.
2. Verify `fraudshield-domain` is still **InService**.
3. **Do NOT delete the domain** -- you will need it for Modules 2, 3, and 4.
4. Studio Domains do not incur charges when idle. Charges only occur when you launch compute resources (notebooks, training jobs, endpoints).

### Step 7 -- Check Billing

1. In the top search bar, type **Billing** and open **AWS Billing and Cost Management**.
2. Click **Bills** in the left navigation.
3. Look for any SageMaker charges for the current day/month.
4. If you cleaned up promptly, charges should be minimal (a few cents for the time the endpoint was running).
5. Note: Charges may take a few hours to appear in the billing dashboard.

---

## Presentation Checkpoint

Be prepared to show:
- The **Inference -> Endpoints** page with no active endpoints
- The **Inference -> Endpoint configurations** and **Models** pages with no leftover resources from this lab
- The **Billing** page (or explain that charges have not appeared yet)
- Explain: Why must you delete the Endpoint before the Endpoint Configuration and Model? (The endpoint is the live resource consuming compute and billing; it also holds a reference to the configuration, which in turn references the model. Deleting in reverse order ensures no dangling references.)
- Explain: What would happen if you forgot to delete the endpoint and left it running overnight? (Continuous charges at the instance rate -- approximately $0.23/hour for `ml.m5.xlarge`, or about $5.50 for 24 hours)

---

## What to Keep for Future Labs

| Resource | Keep? | Why |
|----------|-------|-----|
| Studio Domain (`fraudshield-domain`) | Yes | Needed for Modules 2-4 |
| Default user profile | Yes | Your primary Studio access |
| `analyst-team` profile | Yes | Demonstrates multi-user setup |
| Auto-created execution role | Yes | Needed by the domain |
| `SageMaker-FraudShield-Training` role | Yes | Will use in Module 2 training jobs |
| JumpStart endpoint, config, model | **No -- delete** | Lab is complete |

---

## Key Concepts

- **Cleanup Order:** Endpoint -> Configuration -> Model. Always delete the outermost resource first.
- **Billing Awareness:** SageMaker charges for running compute resources (endpoints, notebook instances, training jobs). Storage (S3, EFS) is minimal. Always check the billing dashboard after labs.
- **Idle vs. Active:** A Studio Domain sitting idle does not generate charges. Only launching compute resources (opening a notebook, creating an endpoint, running a training job) incurs cost.
