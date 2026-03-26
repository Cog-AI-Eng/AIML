# Guide 2: Create a Data Wrangler Flow

This guide walks you through creating a Data Wrangler flow inside SageMaker Studio. You will import the FraudShield transactions CSV from S3, profile the dataset, and apply at least three transformations to prepare features for downstream model training.

---

## Steps

### Step 1 -- Open Studio and Launch Data Wrangler
1. In the SageMaker console, navigate to **Domains** and click your `fraudshield-studio-domain`.
2. Next to the `fraudshield-analyst` user profile, click **Launch** and select **Studio**.
3. Once Studio loads, click **Data** in the left sidebar and select **Data Wrangler**.
4. Click **New flow**. A new `.flow` file opens in the visual editor. This may take a few minutes to provision.

### Step 2 -- Import Data from S3
1. In the Data Wrangler canvas, click **Import data**.
2. Select **Amazon S3** as the data source.
3. Navigate to your bucket and select `raw/ecommerce_transactions.csv`.
4. In the import settings, confirm the file type is **CSV** and the first row is treated as a header.
5. Click **Import** and wait for the preview to load.
6. Verify the column names and row count match your expectations.

### Step 3 -- Profile the Dataset
1. After the import node appears on the canvas, click the **+** icon on the node and choose **Add analysis**.
2. Select **Data Quality and Insights Report** as the analysis type.
3. Set the target column to `is_fraud` (or your label column).
4. Click **Create**. Review the report for missing values, class imbalance, and feature correlations.
5. Note any columns with high null percentages or low predictive power.

### Step 4 -- Add Transform: Handle Missing Values
1. Return to the Data Wrangler canvas and click the **+** icon after the import node.
2. Select **Add transform**.
3. In the transform panel, search for **Handle missing** and select it.
4. Choose **Impute** as the strategy and select **Median** for numeric columns.
5. Apply the transform to all numeric columns with missing values identified in the profile.
6. Click **Preview** to confirm nulls are filled, then click **Add**.

### Step 5 -- Add Transform: Encode Categorical Variables
1. Click the **+** icon after the imputation step and select **Add transform**.
2. Search for **Encode categorical** and select it.
3. Choose **One-hot encoding** as the transform type.
4. Select the categorical column(s) such as `payment_method` or `product_category`.
5. Preview the output to confirm new binary columns are created.
6. Click **Add** to append the step to the flow.

### Step 6 -- Add Transform: Scale Numeric Features
1. Click the **+** after the encoding step and select **Add transform**.
2. Search for **Process numeric** and select **Standard scaler** (zero mean, unit variance).
3. Apply the scaler to continuous columns such as `transaction_amount` and `customer_age`.
4. Preview the transformed values and verify they are centered around zero.
5. Click **Add**.

### Step 7 -- Save and Verify the Flow
1. The flow should now show: **Source --> Import --> Impute --> Encode --> Scale**.
2. Click **File > Save** or press `Ctrl+S` to save the `.flow` file.
3. Click the final transform node and choose **Add analysis > Table summary** to confirm the output schema looks correct.

---

## Presentation Checkpoint
Be prepared to show:
- The full Data Wrangler canvas with all transform nodes visible.
- The data profile report highlighting missing values and class distribution.
- A preview of the final transformed output showing encoded and scaled columns.

---

## Key Concepts
- **Data Wrangler Flow:** A visual, no-code data preparation tool inside SageMaker Studio that records transforms as a directed acyclic graph.
- **Data Profiling:** An automated statistical report that surfaces missing values, outliers, correlations, and class imbalance.
- **Imputation:** Replacing missing values with a statistical proxy (mean, median, or mode) to prevent model training errors.
- **One-Hot Encoding:** Converting categorical variables into binary indicator columns so algorithms that require numeric input can consume them.
- **Standard Scaling:** Centering features to zero mean and unit variance, which improves convergence for distance-based and gradient-based algorithms.
