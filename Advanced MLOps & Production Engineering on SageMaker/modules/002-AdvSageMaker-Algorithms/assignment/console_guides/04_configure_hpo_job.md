# Guide 4: Configure an HPO Job

This guide walks you through creating a Hyperparameter Optimization (HPO) tuning job for the FraudShield XGBoost fraud classifier. You will define the objective metric, configure parameter ranges across continuous, integer, and categorical types, and set up a Bayesian search strategy with 10 trials.

---

## Steps

### Step 1 -- Navigate to Hyperparameter Tuning
1. In the SageMaker console, expand **Training** in the left navigation pane.
2. Click **Hyperparameter tuning jobs**.
3. Click **Create hyperparameter tuning job**.

### Step 2 -- Configure Tuning Job Settings
1. Enter the tuning job name: `fraudshield-xgb-hpo`.
2. Under **Optimization strategy**, select **Bayesian**.
3. Under **Objective metric**, configure:
   - Metric name: `validation:auc`
   - Type: **Maximize**
4. Set **Maximum number of training jobs** to `10`.
5. Set **Maximum number of parallel training jobs** to `2` (this keeps costs manageable).

### Step 3 -- Configure the Training Job Definition
1. Under **Training job definition**, click **Add training job definition**.
2. Enter the definition name: `xgboost-definition`.
3. Under **Algorithm source**, select **SageMaker built-in algorithm** and choose **XGBoost**.
4. Under **IAM role**, select the SageMaker execution role.
5. Under **Resource configuration**, set instance type to `ml.m5.xlarge`, instance count to `1`, volume size to `10` GB.
6. Under **Stopping condition**, set maximum runtime to `3600` seconds per job.

### Step 4 -- Define Hyperparameter Ranges
1. In the **Hyperparameters** section, set the following static (non-tunable) parameters:
   - `objective`: `binary:logistic`
   - `eval_metric`: `auc`
   - `num_round`: `100`
2. Add the following **tunable** parameter ranges:
   - `eta` -- Type: **Continuous**, Min: `0.01`, Max: `0.3`
   - `max_depth` -- Type: **Integer**, Min: `3`, Max: `10`
   - `subsample` -- Type: **Continuous**, Min: `0.5`, Max: `1.0`
   - `colsample_bytree` -- Type: **Continuous**, Min: `0.5`, Max: `1.0`
   - `min_child_weight` -- Type: **Integer**, Min: `1`, Max: `10`
   - `gamma` -- Type: **Continuous**, Min: `0.0`, Max: `5.0`

### Step 5 -- Specify Input and Output
1. Under **Input data configuration**, add a **train** channel:
   - S3 URI: `s3://fraudshield-data-<account-id>/processed/train/`
   - Content type: `text/csv`
2. Add a **validation** channel:
   - S3 URI: `s3://fraudshield-data-<account-id>/processed/validation/`
   - Content type: `text/csv`
3. Under **Output data configuration**, set the S3 path to `s3://fraudshield-data-<account-id>/models/hpo/`.

### Step 6 -- Launch the HPO Job
1. Review all settings: job name, strategy (Bayesian), objective (maximize validation:auc), 10 total trials, 2 parallel.
2. Click **Create hyperparameter tuning job**.
3. The job status changes to **InProgress**. The first two trials begin immediately.
4. Training jobs will launch in batches of 2 until all 10 trials complete. The Bayesian strategy uses results from completed trials to select hyperparameters for subsequent trials.
5. Monitor the job by clicking the tuning job name. The **Training jobs** tab shows each trial's status and objective metric value.

---

## Presentation Checkpoint
Be prepared to show:
- The HPO tuning job configuration page showing Bayesian strategy, objective metric, and parameter ranges.
- The **Training jobs** tab with at least some trials in **Completed** status.
- The range definitions for each tunable hyperparameter.

---

## Key Concepts
- **Hyperparameter Optimization (HPO):** An automated process that searches for the best combination of hyperparameters by running multiple training trials and comparing results against an objective metric.
- **Bayesian Strategy:** A search strategy that builds a probabilistic model of the objective function and uses it to select promising hyperparameter combinations, converging faster than random search.
- **Objective Metric:** The metric the tuning job optimizes. For fraud detection, maximizing AUC ensures the model best separates fraudulent from legitimate transactions.
- **Parameter Ranges:** Continuous ranges sample decimal values, integer ranges sample whole numbers, and categorical ranges sample from a fixed list of options.
