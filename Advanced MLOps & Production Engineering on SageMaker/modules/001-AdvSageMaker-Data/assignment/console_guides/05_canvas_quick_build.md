# Guide 5: Canvas Quick Build

This guide walks you through launching SageMaker Canvas, importing the FraudShield dataset, training a Quick Build model, and examining prediction results. Canvas provides a no-code interface that lets business analysts build ML models without writing any code.

---

## Steps

### Step 1 -- Launch SageMaker Canvas
1. In the SageMaker console, navigate to **Domains** and click your `fraudshield-studio-domain`.
2. Next to the `fraudshield-analyst` user profile, click **Launch** and select **Canvas**.
3. Canvas opens in a new browser tab. The first launch may take several minutes while the application provisions.
4. Once the Canvas home screen loads, you are ready to begin.

### Step 2 -- Import the FraudShield Dataset
1. In Canvas, click **My datasets** in the left sidebar.
2. Click **Import data** and select **Amazon S3** as the source.
3. Navigate to `s3://fraudshield-data-<account-id>/raw/ecommerce_transactions.csv` and select it.
4. Click **Import**. Canvas uploads and previews the dataset.
5. Verify the column names and row count displayed in the preview table.
6. Name the dataset `fraudshield-transactions` if prompted.

### Step 3 -- Create a New Model
1. Click **My models** in the left sidebar and then click **New model**.
2. Enter the model name: `fraudshield-fraud-detector`.
3. Select the problem type **Predictive analysis** (binary classification).
4. Click **Create**.

### Step 4 -- Select the Dataset and Target Column
1. On the model build page, select the `fraudshield-transactions` dataset.
2. Click **Select dataset**.
3. In the column selector, choose `is_fraud` as the **Target column**.
4. Canvas displays a preview of columns that will be used as input features.
5. Review the auto-detected column types and deselect any irrelevant columns (e.g., row IDs or timestamps that should not be features).

### Step 5 -- Run a Quick Build
1. Click **Quick build** (not Standard build) to train a model rapidly.
2. Quick Build trains a sample model in approximately 2-15 minutes.
3. While the model trains, Canvas shows a progress indicator. Do not close the tab.
4. Once training completes, Canvas displays model accuracy metrics on the **Analyze** tab.

### Step 6 -- Examine Predictions and Metrics
1. On the **Analyze** tab, review the overall model accuracy and the column impact chart.
2. Note which features Canvas identified as most predictive of fraud.
3. Click the **Predict** tab.
4. Choose **Single prediction** and enter sample values for a transaction (e.g., high amount, new customer).
5. Observe the predicted label (`is_fraud`: 0 or 1) and the confidence score.
6. Try a second prediction with different values and compare results.

---

## Presentation Checkpoint
Be prepared to show:
- The Canvas model **Analyze** tab with accuracy metrics and column impact.
- A single prediction result showing predicted label and confidence score.
- The dataset preview in Canvas confirming the correct FraudShield data was imported.

---

## Key Concepts
- **SageMaker Canvas:** A no-code ML tool that enables business users to build, train, and generate predictions without writing code or managing infrastructure.
- **Quick Build:** A rapid training mode that uses a subset of the data to produce a model in minutes, suitable for prototyping and feasibility checks.
- **Column Impact:** A Canvas metric that ranks features by their relative contribution to model predictions, similar to feature importance.
- **Binary Classification:** A supervised learning task where the model predicts one of two discrete outcomes (in this case, fraud or not fraud).
