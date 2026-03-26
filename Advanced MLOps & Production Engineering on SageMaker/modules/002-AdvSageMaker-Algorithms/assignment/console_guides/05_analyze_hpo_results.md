# Guide 5: Analyze HPO Results

This guide walks you through navigating the HPO tuning job results in the SageMaker console. You will identify the best trial, compare hyperparameter values across runs, and extract the winning configuration for production deployment.

---

## Steps

### Step 1 -- Open the Tuning Job
1. In the SageMaker console, expand **Training** and click **Hyperparameter tuning jobs**.
2. Click the tuning job `fraudshield-xgb-hpo`.
3. Verify the overall status is **Completed** (or wait until all 10 trials finish).

### Step 2 -- Identify the Best Training Job
1. Click the **Best training job** tab.
2. This tab highlights the trial that achieved the highest `validation:auc` value.
3. Note the best trial's training job name, objective metric value, and hyperparameter values.
4. Record these values -- they represent the optimal configuration found by the Bayesian search.

### Step 3 -- Compare All Trials
1. Click the **Training jobs** tab to see all 10 trials listed in a table.
2. The table shows each trial's status, objective metric value, and hyperparameter settings.
3. Sort by the **Objective metric value** column (descending) to rank trials from best to worst.
4. Compare the top 3 trials. Note how the hyperparameter values differ and whether certain ranges consistently appear in top-performing trials.
5. Look for patterns: for example, do the best trials share similar `max_depth` or `eta` values?

### Step 4 -- Review Training Curves
1. Click on the best training job name to open its detail page.
2. Scroll to the **Monitor** section and examine the CloudWatch training curves.
3. Look at the `train:auc` and `validation:auc` curves. Confirm they converge and that the validation metric does not diverge (which would indicate overfitting).
4. Go back and click the worst-performing trial for comparison. Note differences in convergence behavior.

### Step 5 -- Export the Best Configuration
1. Return to the **Best training job** tab.
2. Record the following values for the winning trial:
   - `eta`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`
   - Final `validation:auc` value
   - Training job ARN
   - Model artifact S3 path
3. These values will be used to create the production training job or to register the model in the Model Registry (covered in a later module).

### Step 6 -- Review Tuning Job Analytics
1. On the tuning job page, check the **Tuning job analytics** section if available.
2. This visualization shows how the objective metric improved across trials over time.
3. In a Bayesian search, early trials explore the parameter space broadly, while later trials exploit promising regions. Confirm this pattern by observing that later trials generally achieve higher AUC values.

---

## Presentation Checkpoint
Be prepared to show:
- The **Best training job** tab with the winning trial's hyperparameters and AUC value.
- The sorted **Training jobs** table comparing all 10 trials.
- The training curve for the best trial showing train and validation AUC convergence.

---

## Key Concepts
- **Best Training Job:** The trial within an HPO job that achieved the highest (or lowest, depending on objective) metric value. SageMaker surfaces this automatically.
- **Bayesian Convergence:** As trials progress, the Bayesian optimizer narrows the search space around high-performing regions, producing increasingly better results.
- **Overfitting Detection:** When train AUC climbs but validation AUC plateaus or drops, the model is memorizing training data rather than learning generalizable patterns.
- **HPO Analytics:** A visualization that plots objective metric values across sequential trials, illustrating how the optimizer explores and then exploits the hyperparameter space.
