# Guide 1: Configure Spot Training

Launch a FraudShield training job with Managed Spot Instances enabled and checkpointing configured. Spot training can reduce training costs by up to 90% compared to on-demand instances by using spare AWS capacity.

---

## Steps

### Step 1 -- Navigate to Training Jobs

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar, expand **Training** and select **Training jobs**.
3. Click **Create training job**.

---

### Step 2 -- Configure the Training Job Basics

1. For **Job name**, enter `ASM-FraudShield-SpotTraining`.
2. Under **IAM role**, select your SageMaker execution role.
3. Under **Algorithm options**, select **Built-in algorithm** and choose **XGBoost**.
4. For **Algorithm version**, select the latest available (e.g., 1.5-1).
5. Under **Resource configuration**:
   - **Instance type:** Select `ml.m5.xlarge`.
   - **Instance count:** `1`.
   - **Additional storage volume per instance (GB):** `20`.

---

### Step 3 -- Enable Managed Spot Training

1. Under **Managed spot training**, toggle the setting to **Enabled**.
2. Once enabled, two new fields appear:
   - **Maximum runtime (seconds):** Enter `3600` (1 hour). This is the maximum wall-clock time for the entire job including interruptions.
   - **Maximum wait time (seconds):** Enter `7200` (2 hours). This is the maximum time SageMaker will wait for spot capacity before falling back or failing.
3. The maximum wait time must be greater than or equal to the maximum runtime.

---

### Step 4 -- Configure Checkpointing

1. Under **Checkpoint configuration**:
   - **S3 checkpoint path:** Enter `s3://sagemaker-fraudshield-<account-id>/checkpoints/spot-training/`.
   - **Local checkpoint path:** Enter `/opt/ml/checkpoints`.
2. Checkpointing allows the training job to resume from the last saved state if a spot interruption occurs, rather than restarting from scratch.

---

### Step 5 -- Configure Input and Output Channels

1. Under **Input data configuration**, click **Add channel**:
   - **Channel name:** `train`
   - **S3 data type:** `S3Prefix`
   - **S3 URI:** `s3://sagemaker-fraudshield-<account-id>/training-data/train/`
   - **Content type:** `text/csv`
2. Add another channel for validation (optional):
   - **Channel name:** `validation`
   - **S3 URI:** `s3://sagemaker-fraudshield-<account-id>/training-data/validation/`
3. Under **Output data configuration**:
   - **S3 output path:** `s3://sagemaker-fraudshield-<account-id>/models/spot-training/`
4. Set hyperparameters as needed (e.g., `num_round=100`, `objective=binary:logistic`, `eval_metric=auc`).
5. Click **Create training job**.

---

### Step 6 -- Compare Spot vs On-Demand Cost

1. Wait for the training job to complete (status changes to **Completed**).
2. Click on the job name to open the detail page.
3. Under **Job details**, locate the following fields:
   - **Training seconds:** The actual compute time used.
   - **Billable seconds:** The seconds you are charged for (reduced with spot).
   - **Managed Spot Training savings:** The percentage saved compared to on-demand pricing.
4. Document the savings percentage. Spot training typically saves 60-90% depending on availability.

---

## Presentation Checkpoint
Be prepared to show:
- The training job detail page with **ManagedSpotTrainingEnabled** set to true
- The checkpoint S3 path configuration
- The **Managed Spot Training savings** percentage after job completion

---

## Key Concepts
- **Managed Spot Instances:** AWS provisions training compute from spare capacity at a steep discount. The trade-off is that instances can be interrupted with 2 minutes of notice.
- **Checkpointing:** Periodically saves training state to S3 so that if an interruption occurs, the job resumes from the last checkpoint rather than starting over.
- **Maximum Wait Time:** The total time (including interruptions and waiting for new capacity) that SageMaker allows before declaring the job failed. Must exceed the expected training time.
- **Billable Seconds:** With spot training, you are billed only for actual compute time, not time spent waiting for capacity.
