# Guide 4: Configure Model Quality Monitoring

Create a model quality baseline with evaluation metrics and configure model quality monitoring that compares ongoing model performance against the baseline using ground truth labels. This detects model accuracy degradation over time.

---

## Steps

### Step 1 -- Prepare Ground Truth Data

1. Open the **AWS Management Console** and navigate to **Amazon S3**.
2. Open your bucket `sagemaker-fraudshield-<account-id>`.
3. Create a folder named `ground-truth/`.
4. Create a CSV file named `ground_truth_labels.csv` with two columns: `inference_id` and `label`. The `inference_id` should correspond to the inference IDs from your captured data, and `label` should be the actual fraud indicator (0 or 1).
5. Upload this file to `s3://sagemaker-fraudshield-<account-id>/ground-truth/ground_truth_labels.csv`.

---

### Step 2 -- Create a Model Quality Baseline

1. Navigate to **Amazon SageMaker** and select **Model Monitor** under **Inference**.
2. Navigate to the **Model quality** tab.
3. Click **Create baseline job**.
4. For **Job name**, enter `ASM-FraudShield-MQ-Baseline`.
5. Under **Problem type**, select **Binary classification**.
6. Under **Baseline dataset**:
   - Provide a dataset that includes both predictions and ground truth labels so SageMaker can compute baseline evaluation metrics.
   - **S3 input location:** `s3://sagemaker-fraudshield-<account-id>/baseline-data/training_predictions.csv`
   - This file should contain columns for predictions and actual labels.
7. Configure the column mapping:
   - **Inference attribute:** Column index or name containing the predicted value.
   - **Ground truth attribute:** Column index or name containing the actual label.
   - **Probability attribute:** Column containing the probability score (if applicable).
   - **Probability threshold:** `0.5`.
8. Under **Output configuration**:
   - **S3 output path:** `s3://sagemaker-fraudshield-<account-id>/baseline-output/model-quality/`
9. Under **Compute configuration**:
   - **Instance type:** `ml.m5.xlarge`.
   - **Instance count:** `1`.
10. Click **Create baseline job** and wait for it to complete.

---

### Step 3 -- Examine Baseline Metrics

1. Once the baseline job completes, navigate to S3 at `baseline-output/model-quality/`.
2. Open the output files. The baseline produces metrics such as:
   - Accuracy, Precision, Recall, F1 Score
   - AUC-ROC
   - Confusion matrix values
3. Note the baseline accuracy and F1 score -- these are the benchmarks for ongoing monitoring.

---

### Step 4 -- Create a Model Quality Monitoring Schedule

1. Return to the **Model Monitor** section in SageMaker.
2. Under the **Model quality** tab, click **Create monitoring schedule**.
3. For **Schedule name**, enter `ASM-FraudShield-MQ-Schedule`.
4. Under **Endpoint**, select your FraudShield endpoint.
5. Under **Problem type**, select **Binary classification**.
6. Under **Ground truth input**:
   - **S3 URI:** `s3://sagemaker-fraudshield-<account-id>/ground-truth/`
   - This is where you will periodically upload ground truth labels to be merged with captured predictions.
7. Under **Baseline**:
   - Reference the statistics and constraints files from the model quality baseline output.
8. Under **Schedule**, select **Hourly**.
9. Under **Output configuration**:
   - **S3 output path:** `s3://sagemaker-fraudshield-<account-id>/monitoring-output/model-quality/`
10. Under **Compute configuration**, select `ml.m5.xlarge` with 1 instance.
11. Click **Create monitoring schedule**.

---

### Step 5 -- Examine a Model Quality Report

1. After the next monitoring execution completes (may require waiting for the hourly trigger and having ground truth data available), navigate to S3.
2. Open `monitoring-output/model-quality/` and find the latest execution folder.
3. Review the report for metrics like accuracy, precision, recall, and any deviations from the baseline.
4. If ground truth was not available for the window, the execution may report insufficient data.

---

## Presentation Checkpoint
Be prepared to show:
- The completed model quality baseline job with evaluation metrics
- The model quality monitoring schedule configuration showing ground truth S3 path
- The baseline metrics (accuracy, F1, AUC) from the baseline output

---

## Key Concepts
- **Model Quality Monitoring:** Tracks how well the model's predictions match actual outcomes over time. Unlike data quality monitoring which checks inputs, model quality monitoring checks prediction accuracy.
- **Ground Truth:** The actual labels (e.g., whether a transaction was truly fraudulent) that are uploaded after the fact. Without ground truth, model quality cannot be assessed.
- **Binary Classification Metrics:** For fraud detection, key metrics include precision (false positive rate), recall (missed fraud rate), F1 score (harmonic mean), and AUC-ROC.
- **Delayed Ground Truth:** In practice, ground truth arrives with a delay (days or weeks after the prediction). The monitoring schedule merges ground truth with captured predictions when both are available.
