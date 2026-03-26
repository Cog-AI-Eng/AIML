# Guide 2: Right-Size Instances

Launch a training job on ml.m5.xlarge, observe CloudWatch CPU and memory metrics during and after training, and analyze whether the instance is appropriately sized. Document a right-sizing recommendation based on actual utilization data.

---

## Steps

### Step 1 -- Launch a Training Job for Observation

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar, select **Training jobs** and click **Create training job**.
3. For **Job name**, enter `ASM-FraudShield-RightSizing`.
4. Configure the job using the same settings as Guide 1:
   - **Algorithm:** XGBoost (built-in).
   - **Instance type:** `ml.m5.xlarge`.
   - **Instance count:** `1`.
   - **Input channel:** Your FraudShield training data in S3.
   - **Output path:** `s3://sagemaker-fraudshield-<account-id>/models/right-sizing/`
5. Leave Managed Spot Training **disabled** for this job (so metrics are uninterrupted).
6. Click **Create training job** and wait for it to complete.

---

### Step 2 -- Navigate to CloudWatch Metrics

1. Navigate to **Amazon CloudWatch** in the console.
2. In the left sidebar, select **Metrics** then **All metrics**.
3. In the metrics browser, search for `/aws/sagemaker/TrainingJobs`.
4. Select the **Host** dimension group.
5. Filter by the training job name `ASM-FraudShield-RightSizing`.

---

### Step 3 -- Analyze CPU Utilization

1. Select the `CPUUtilization` metric for your training job.
2. Set the **Period** to `1 minute` and the **Statistic** to `Average`.
3. Expand the time range to cover the full training job duration.
4. Observe the CPU utilization graph:
   - **Under 30% average:** The instance is over-provisioned. A smaller instance (e.g., `ml.m5.large`) would suffice.
   - **50-80% average:** The instance is appropriately sized.
   - **Over 90% sustained:** The instance may be a bottleneck. Consider a larger instance.
5. Note the peak and average CPU utilization values.

---

### Step 4 -- Analyze Memory Utilization

1. In the same metrics view, select the `MemoryUtilization` metric for your training job.
2. Set the **Period** to `1 minute` and **Statistic** to `Average`.
3. Observe the memory utilization graph:
   - **Under 30% average:** The instance has more memory than needed.
   - **Over 85% sustained:** The job is memory-constrained and may benefit from a memory-optimized instance.
4. Note the peak and average memory utilization values.

---

### Step 5 -- Analyze Disk and Network (Optional)

1. Check for `DiskUtilization` metric if available.
2. Check for `NetworkIn` and `NetworkOut` metrics to see data transfer volumes during training.
3. High network I/O with low CPU may indicate the job is I/O-bound rather than compute-bound.

---

### Step 6 -- Document Your Right-Sizing Recommendation

1. Based on your observations, write a recommendation in the following format:
   - **Current instance:** ml.m5.xlarge (4 vCPUs, 16 GB RAM)
   - **Average CPU utilization:** __%
   - **Peak CPU utilization:** __%
   - **Average memory utilization:** __%
   - **Peak memory utilization:** __%
   - **Recommendation:** [Keep / Downsize to ml.m5.large / Upsize to ml.m5.2xlarge]
   - **Rationale:** [1-2 sentences explaining why]
2. Save this recommendation -- you will present it as a deliverable.

---

## Presentation Checkpoint
Be prepared to show:
- CloudWatch graphs showing CPU utilization over the training job duration
- CloudWatch graphs showing memory utilization over the training job duration
- Your written right-sizing recommendation with data points

---

## Key Concepts
- **Right-Sizing:** The practice of matching instance type to actual workload requirements. Over-provisioning wastes money; under-provisioning causes bottlenecks or failures.
- **CPU Utilization:** Measures compute usage. XGBoost on a single instance should show moderate to high CPU usage. Very low utilization suggests the instance is larger than needed.
- **Memory Utilization:** Measures RAM usage. If the dataset fits comfortably in memory with headroom to spare, a smaller instance may work.
- **ml.m5 Instance Family:** General-purpose instances balanced between compute, memory, and networking. The xlarge size provides 4 vCPUs and 16 GB RAM.
