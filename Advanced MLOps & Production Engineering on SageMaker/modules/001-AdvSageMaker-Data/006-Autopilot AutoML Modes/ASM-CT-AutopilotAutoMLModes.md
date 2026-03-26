# Autopilot AutoML Modes

**Estimated Time:** 10 Minutes

## Introduction

In the previous topic, you saw that Canvas's Standard Build mode uses SageMaker Autopilot under the hood. Autopilot is SageMaker's AutoML service: given a tabular dataset and a target column, it automatically explores data preprocessing strategies, algorithm families, and hyperparameter configurations, then returns a ranked leaderboard of candidate models with full explainability reports.

What makes Autopilot distinct from other AutoML tools is transparency. Every pipeline Autopilot generates is available as editable Python code. You can inspect exactly which preprocessing steps it chose, which algorithm it selected, and which hyperparameters it tuned. This makes Autopilot useful not only as a "build a model for me" tool but also as a structured algorithm selection framework -- you can run Autopilot, study its best pipeline, and then implement your own refined version in a training script.

This reading covers Autopilot's two training modes (Ensembling and HPO), how to launch jobs from the console, and how to interpret the output artifacts.

## Core Concepts

### Launching an Autopilot job from the console

1. Navigate to **SageMaker > AutoML** in the console sidebar.
2. Click **Create experiment**.
3. **Experiment name:** Enter a descriptive name (e.g., `fraud-detection-autopilot`).
4. **Input data:** Specify the S3 URI of your CSV dataset. Autopilot reads the schema automatically.
5. **Target column:** Select the column you want to predict.
6. **Problem type:** Autopilot auto-detects (binary classification, multiclass, or regression), but you can override.
7. **Training mode:** Select **Ensembling** or **HPO** (detailed below).
8. **Output location:** Specify an S3 URI for model artifacts and generated notebooks.
9. Click **Create experiment**. Autopilot begins processing.

### Ensembling mode

In Ensembling mode, Autopilot trains multiple algorithms on the same data and combines their predictions using a stacking ensemble. The process:

1. **Data analysis:** Autopilot profiles the dataset, identifies column types, detects missing values, and determines preprocessing strategies.
2. **Feature engineering:** Applies automatic transformations (encoding, scaling, missing value imputation) tailored to each algorithm family.
3. **Multi-algorithm training:** Trains models from several algorithm families simultaneously (typically XGBoost, linear learner, and deep learning models).
4. **Stacking:** Trains a meta-learner that combines the base model predictions to produce a single final prediction.
5. **Leaderboard:** Ranks all individual models and the final ensemble by the objective metric.

Ensembling mode typically produces the highest-accuracy models because the meta-learner can capture complementary strengths of different algorithms. The trade-off is that the final model is larger (it contains multiple base models plus the meta-learner) and inference latency is higher.

**Best for:** Maximizing predictive accuracy when inference latency is not a constraint (e.g., batch scoring use cases).

### HPO mode (Hyperparameter Optimization)

In HPO mode, Autopilot selects the single best algorithm family for your data and then runs a systematic hyperparameter optimization search to find the best configuration for that algorithm. The process:

1. **Data analysis and feature engineering:** Same as Ensembling mode.
2. **Algorithm selection:** Autopilot evaluates candidate algorithms on a sample and selects the most promising family.
3. **HPO search:** Runs multiple training jobs with different hyperparameter configurations using Bayesian optimization. Each job is a separate SageMaker Training Job with its own instance.
4. **Leaderboard:** Ranks all HPO trial models by the objective metric.

HPO mode produces a single model (not an ensemble), which means smaller model artifacts and faster inference. The accuracy may be slightly lower than the Ensembling mode's stacked model, but for many use cases the difference is negligible.

**Best for:** Production endpoints where inference latency matters, or when you want a single interpretable model rather than an opaque stack.

### Understanding the output

After an Autopilot job completes, you can examine the results in the console at **SageMaker > AutoML > [your experiment]**:

- **Leaderboard:** A ranked table of candidate models with metrics (accuracy, F1, RMSE, etc.), training time, and instance type used.
- **Best model:** The top-ranked model, deployable to an endpoint with one click.
- **Generated notebooks:** Autopilot produces two notebooks:
  - **Data Exploration Notebook:** Contains the data analysis and profiling Autopilot performed, including visualizations of feature distributions and correlation analysis.
  - **Candidate Generation Notebook:** Contains the complete Python code for every preprocessing pipeline and model training configuration Autopilot evaluated. You can download this notebook, modify it, and run it yourself.
- **Explainability report:** If enabled, Autopilot generates SHAP-based feature importance for the best model, showing which features drive predictions.

### Deploying an Autopilot model

From the leaderboard, select the best model and click **Deploy**. The deployment dialog asks for:

- **Endpoint name**
- **Instance type** (use `ml.m5.xlarge` to stay within Free Tier limits)
- **Instance count**

After deployment, the model is available as a real-time inference endpoint, identical to endpoints you created manually in the foundational SageMaker skill. You invoke it the same way: `boto3` `invoke_endpoint` with a CSV or JSON payload.

### Autopilot vs. manual training

Autopilot is not a replacement for understanding ML fundamentals. Its value lies in:

- **Baselining:** Run Autopilot first to establish an accuracy benchmark. Then build your own model and compare.
- **Algorithm discovery:** Study the generated Candidate Notebook to see which algorithms and preprocessing strategies Autopilot found effective for your data. This is a concrete, data-driven version of the Algorithm Selection Framework from the AIML skill.
- **Rapid prototyping:** Get a deployable model in hours rather than days, useful for proof-of-concept work.

For production systems, most teams start with Autopilot, study the output, and then build a custom training pipeline informed by Autopilot's findings.

## Connecting to Practice

This topic gives you the technical understanding of what Canvas's Standard Build is doing behind the scenes. The next topic, *Canvas vs. Autopilot*, compares the two approaches head-to-head. In the module lecture, you will launch an Autopilot job from the console, examine the generated notebooks, and deploy the best model. The assignment will require you to use Autopilot's output as a baseline for a custom training approach in Module 2.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Autopilot Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html)** - *Docs*: Complete reference for Autopilot job configuration, training modes, and output interpretation.
- **[Autopilot Model Explainability](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-model-support-validation.html)** - *Docs*: Details on the SHAP-based explainability reports Autopilot generates.

**Interactive practice**

- **[SageMaker Autopilot Workshop](https://catalog.workshops.aws/sagemaker-autopilot/en-US)** - *Interactive*: End-to-end lab covering Autopilot job creation, leaderboard analysis, and model deployment.
