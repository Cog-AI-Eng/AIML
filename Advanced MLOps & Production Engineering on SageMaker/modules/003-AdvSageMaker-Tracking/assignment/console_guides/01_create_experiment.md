# Guide 1: Create an Experiment

This guide walks you through creating a SageMaker Experiment and running three XGBoost training jobs with different hyperparameter configurations, each associated with the experiment. This establishes the foundation for organized tracking and comparison of model iterations.

---

## Steps

### Step 1 -- Open SageMaker Studio
1. In the SageMaker console, navigate to **Domains** and click your Studio domain.
2. Launch **Studio** for your user profile.
3. Once Studio loads, click **Home** in the left sidebar and then select **Experiments**.

### Step 2 -- Create an Experiment
1. On the Experiments page, click **Create experiment**.
2. Enter the experiment name: `fraudshield-fraud-detection`.
3. Optionally enter a description: `Comparison of XGBoost hyperparameter configurations for FraudShield fraud classification`.
4. Click **Create experiment**. The experiment appears in the list with zero runs.

### Step 3 -- Launch Training Job 1 (Conservative)
1. Open a new browser tab and navigate to the SageMaker console.
2. Go to **Training > Training jobs** and click **Create training job**.
3. Enter the job name: `fraudshield-exp-run1-conservative`.
4. Under **Algorithm source**, select **SageMaker built-in algorithm** and choose **XGBoost**.
5. Under **Hyperparameters**, configure:
   - `objective`: `binary:logistic`
   - `eval_metric`: `auc`
   - `num_round`: `100`
   - `max_depth`: `3`
   - `eta`: `0.1`
   - `subsample`: `0.8`
6. Under **Experiment configuration**, select the experiment `fraudshield-fraud-detection` and enter run name: `run1-conservative`.
7. Set instance type to `ml.m5.xlarge`, instance count `1`, volume `10` GB.
8. Add train and validation channels pointing to your S3 data paths. Set output to `s3://fraudshield-data-<account-id>/models/experiments/run1/`.
9. Click **Create training job**.

### Step 4 -- Launch Training Job 2 (Moderate)
1. Click **Create training job** again.
2. Enter the job name: `fraudshield-exp-run2-moderate`.
3. Select XGBoost and configure hyperparameters:
   - `objective`: `binary:logistic`
   - `eval_metric`: `auc`
   - `num_round`: `150`
   - `max_depth`: `5`
   - `eta`: `0.2`
   - `subsample`: `0.7`
4. Under **Experiment configuration**, select `fraudshield-fraud-detection` and enter run name: `run2-moderate`.
5. Use the same resource and input configuration as Run 1. Set output to `.../experiments/run2/`.
6. Click **Create training job**.

### Step 5 -- Launch Training Job 3 (Aggressive)
1. Click **Create training job** once more.
2. Enter the job name: `fraudshield-exp-run3-aggressive`.
3. Select XGBoost and configure hyperparameters:
   - `objective`: `binary:logistic`
   - `eval_metric`: `auc`
   - `num_round`: `200`
   - `max_depth`: `8`
   - `eta`: `0.3`
   - `subsample`: `0.6`
4. Under **Experiment configuration**, select `fraudshield-fraud-detection` and enter run name: `run3-aggressive`.
5. Use the same resource and input configuration. Set output to `.../experiments/run3/`.
6. Click **Create training job**.

### Step 6 -- Verify Runs in the Experiment
1. Return to SageMaker Studio and navigate to **Experiments**.
2. Click `fraudshield-fraud-detection`. You should see three runs listed.
3. Wait for all three training jobs to reach **Completed** status. Each job should take 3-8 minutes.
4. Confirm that each run shows its associated training job name and hyperparameter values.

---

## Presentation Checkpoint
Be prepared to show:
- The experiment `fraudshield-fraud-detection` with three runs listed in Studio.
- The hyperparameter differences across the three runs (conservative, moderate, aggressive).
- All three training jobs in **Completed** status.

---

## Key Concepts
- **SageMaker Experiment:** A named container that groups related training runs (trials) for organized tracking, comparison, and auditing.
- **Run (Trial):** A single training execution associated with an experiment. Each run records hyperparameters, metrics, and artifacts.
- **Experiment Configuration:** A section in the training job creation form that links the job to an experiment and assigns a run name.
- **Hyperparameter Variation:** Running the same algorithm with different hyperparameter settings allows systematic comparison of model performance.
