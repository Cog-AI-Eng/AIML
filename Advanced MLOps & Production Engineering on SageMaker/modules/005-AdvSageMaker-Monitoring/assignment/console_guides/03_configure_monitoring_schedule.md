# Guide 3: Configure a Data Quality Monitoring Schedule

Create a recurring data quality monitoring schedule that automatically compares captured inference data against your baseline. When incoming data drifts from expected patterns, the monitor generates a violation report in S3.

---

## Steps

### Step 1 -- Navigate to Monitoring Schedules

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar under **Inference**, select **Model Monitor**.
3. Navigate to the **Monitoring schedules** tab (or **Data quality** tab depending on console version).
4. Click **Create monitoring schedule**.

---

### Step 2 -- Configure the Monitoring Schedule

1. For **Schedule name**, enter `ASM-FraudShield-DQ-Schedule`.
2. Under **Endpoint**, select your FraudShield endpoint (the one with data capture enabled from Guide 1).
3. Under **Schedule**:
   - Select **Hourly** as the monitoring frequency.
   - This means a monitoring job will run every hour to analyze captured data from the preceding hour.
4. Under **Baseline** (or **Constraints and statistics**):
   - **Statistics file:** `s3://sagemaker-fraudshield-<account-id>/baseline-output/data-quality/statistics.json`
   - **Constraints file:** `s3://sagemaker-fraudshield-<account-id>/baseline-output/data-quality/constraints.json`
5. Under **Output configuration**:
   - **S3 output path:** `s3://sagemaker-fraudshield-<account-id>/monitoring-output/data-quality/`

---

### Step 3 -- Configure Compute and Role

1. Under **Compute configuration**:
   - **Instance type:** Select `ml.m5.xlarge`.
   - **Instance count:** `1`.
   - **Volume size (GB):** `20`.
2. Under **IAM role**, select your SageMaker execution role.
3. Review all settings and click **Create monitoring schedule**.

---

### Step 4 -- Generate Traffic and Wait for an Execution

1. The schedule is now active. To trigger a monitoring execution, the endpoint needs captured data from the current hour.
2. Open **AWS CloudShell** and send additional invocations to generate fresh captured data:
   ```bash
   for i in $(seq 1 20); do
     aws sagemaker-runtime invoke-endpoint \
       --endpoint-name <your-endpoint-name> \
       --content-type text/csv \
       --body "50.0,1,0,1,234.56,2,0.85,1200,$i" \
       --region us-east-1 \
       /dev/null
   done
   ```
3. Wait for the next hourly monitoring execution to trigger (check at the top of the next hour).
4. Return to the **Monitoring schedules** page and click on your schedule name.
5. Under **Monitoring executions**, look for a new execution with status **Completed**.

---

### Step 5 -- Examine the Violation Report

1. Click on the completed execution to view its details.
2. Note the execution status and any detected violations.
3. Navigate to **Amazon S3** and go to `monitoring-output/data-quality/`.
4. Open the execution-specific subfolder (organized by timestamp).
5. Locate the `constraint_violations.json` file. Open it and examine:
   - Which features violated constraints.
   - The type of violation (data type mismatch, value out of range, completeness drop).
6. If no violations were detected (because test data closely matches training data), the violations file will be empty or absent, which indicates a healthy state.

---

## Presentation Checkpoint
Be prepared to show:
- The monitoring schedule configuration page showing hourly frequency and baseline references
- At least one completed monitoring execution in the execution history
- The violation report (or confirmation of a clean report) in S3

---

## Key Concepts
- **Monitoring Schedule:** An automated, recurring job that compares recent inference data against the baseline. SageMaker manages the scheduling, compute provisioning, and teardown.
- **Hourly Cadence:** Each execution analyzes data captured since the last execution. Hourly is the finest granularity; you can also schedule daily or custom cron expressions.
- **Constraint Violations:** When incoming data deviates from baseline constraints (e.g., a new category appears, a numeric column has nulls), the monitor logs the violation with details about which constraint failed.
- **No-Data Executions:** If no data was captured during the monitoring window, the execution completes with a status indicating no data was available to analyze.
