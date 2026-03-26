# Guide 2: Train K-Means Clustering

This guide walks you through launching a K-Means clustering training job on FraudShield customer features. K-Means groups customers into behavioral segments, which can reveal spending patterns and risk profiles without requiring labeled data.

---

## Steps

### Step 1 -- Prepare Customer Feature Data
1. Before creating the training job, ensure your customer feature file is in S3. This file should contain numeric columns only (e.g., `transaction_amount`, `customer_age`, `avg_order_value`, `num_transactions`).
2. The file should be in CSV format with no header row (SageMaker's built-in K-Means expects headerless CSV or RecordIO).
3. Verify the file is at `s3://fraudshield-data-<account-id>/processed/clustering/customer_features.csv`.

### Step 2 -- Create the Training Job
1. In the SageMaker console, navigate to **Training > Training jobs**.
2. Click **Create training job**.
3. Enter the job name: `fraudshield-kmeans-segments`.
4. Under **IAM role**, select the SageMaker execution role.
5. Under **Algorithm source**, select **SageMaker built-in algorithm**.
6. From the list, select **K-Means**.

### Step 3 -- Set Hyperparameters
1. In the **Hyperparameters** section, configure:
   - `k`: `5` (number of clusters -- start with 5 customer segments)
   - `feature_dim`: enter the number of feature columns in your dataset (e.g., `4`)
   - `mini_batch_size`: `500`
   - `epochs`: `10`
2. Leave other parameters at defaults.

### Step 4 -- Configure Resources and Input
1. Under **Resource configuration**, set instance type to `ml.m5.xlarge`, instance count to `1`, and volume size to `10` GB.
2. Under **Stopping condition**, set maximum runtime to `3600` seconds.
3. Under **Input data configuration**, add a **train** channel:
   - S3 URI: `s3://fraudshield-data-<account-id>/processed/clustering/`
   - Content type: `text/csv`
   - S3 data type: **S3Prefix**
4. Under **Output data configuration**, set the output path to `s3://fraudshield-data-<account-id>/models/kmeans/`.

### Step 5 -- Launch and Monitor
1. Click **Create training job**.
2. Wait for the job to reach **Completed** status. K-Means typically completes in 2-5 minutes on moderate datasets.
3. Observe the training metrics in the **Monitor** section. Look for `train:msd` (mean squared distance), which should decrease across epochs.

### Step 6 -- Examine the Model Artifact
1. Once the job completes, note the S3 path to the output `model.tar.gz`.
2. The K-Means model artifact contains the cluster centroids. You can optionally deploy this model to a real-time endpoint (covered in a later module) to assign new customers to clusters.
3. Review the final `train:msd` value to assess cluster compactness.

---

## Presentation Checkpoint
Be prepared to show:
- The completed K-Means training job with status **Completed**.
- The hyperparameters, particularly `k=5` and `feature_dim`.
- The `train:msd` metric from the CloudWatch training chart, showing convergence across epochs.

---

## Key Concepts
- **K-Means Clustering:** An unsupervised algorithm that partitions data into k groups by minimizing the distance between data points and their assigned cluster centroid.
- **feature_dim:** A required hyperparameter for SageMaker's K-Means that specifies the number of input features. This must match the column count in your dataset.
- **Mean Squared Distance (MSD):** The average squared distance between each data point and its assigned centroid. Lower MSD indicates tighter, more cohesive clusters.
- **Unsupervised Learning:** Training without labeled target values. The algorithm discovers structure in the data rather than learning a mapping from inputs to outputs.
