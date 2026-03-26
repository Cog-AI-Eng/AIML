# Guide 2: Create a Data Quality Baseline

Run a baseline job on the FraudShield training data to generate statistical profiles and constraint rules. These baselines define what "normal" data looks like and serve as the reference point for all future data quality monitoring.

---

## Steps

### Step 1 -- Upload the Training Dataset

1. Open the **AWS Management Console** and navigate to **Amazon S3**.
2. Open your bucket `sagemaker-fraudshield-<account-id>`.
3. Create a folder named `baseline-data/`.
4. Upload your FraudShield training dataset as a CSV file (with headers) named `training_data.csv`.
5. Confirm it is available at `s3://sagemaker-fraudshield-<account-id>/baseline-data/training_data.csv`.

---

### Step 2 -- Navigate to Model Monitor

1. Navigate to **Amazon SageMaker** in the console.
2. In the left sidebar, expand **Inference** and select **Model Monitor**.
3. If this is your first time, you will see a landing page explaining Model Monitor capabilities. Click through to the monitoring section.

---

### Step 3 -- Create a Data Quality Baseline Job

1. In the Model Monitor section, navigate to the **Data quality** tab.
2. Click **Create baseline job** (or **Suggest baseline**).
3. For **Job name**, enter `ASM-FraudShield-DQ-Baseline`.
4. Under **Baseline dataset**:
   - **S3 input location:** `s3://sagemaker-fraudshield-<account-id>/baseline-data/training_data.csv`
   - **Dataset format:** Select `CSV`.
   - **Header:** Select `Yes` if your CSV includes column headers.
5. Under **Output configuration**:
   - **S3 output location:** `s3://sagemaker-fraudshield-<account-id>/baseline-output/data-quality/`
6. Under **Compute configuration**:
   - **Instance type:** Select `ml.m5.xlarge`.
   - **Instance count:** `1`.
   - **Volume size (GB):** `20`.
7. Under **IAM role**, select your SageMaker execution role.
8. Click **Create baseline job**.

---

### Step 4 -- Monitor the Baseline Job

1. The job will appear in the baseline jobs list with status **InProgress**.
2. Click on the job name to view its detail page.
3. Wait for the job to complete (typically 5-10 minutes depending on dataset size).
4. When the status changes to **Completed**, proceed to examine the outputs.

---

### Step 5 -- Examine the Baseline Output Files

1. Navigate to **Amazon S3** and open your bucket.
2. Go to `baseline-output/data-quality/`.
3. Locate two key output files:
   - **statistics.json** -- Contains computed statistics for each feature: mean, standard deviation, min, max, distribution summaries, and missing value counts.
   - **constraints.json** -- Contains auto-generated constraint rules for each feature: data types, non-null expectations, and value range boundaries.
4. Download and open `statistics.json`. Identify the mean and standard deviation for at least two features.
5. Download and open `constraints.json`. Identify the data type and completeness constraints for each column.

---

## Presentation Checkpoint
Be prepared to show:
- The completed baseline job in the SageMaker console
- The `statistics.json` file with computed feature statistics
- The `constraints.json` file with auto-generated constraint rules

---

## Key Concepts
- **Baseline:** A statistical profile of your training data that defines expected feature distributions, data types, and completeness. It serves as the reference for detecting drift.
- **statistics.json:** Contains descriptive statistics (mean, median, std, quantiles, unique counts) for every feature in the dataset.
- **constraints.json:** Contains rules that incoming data should satisfy, such as expected data types, non-null fractions, and value ranges. These rules are automatically inferred from the baseline data.
- **Drift Detection:** During monitoring, new data is compared against these baseline statistics and constraints. Deviations are reported as violations.
