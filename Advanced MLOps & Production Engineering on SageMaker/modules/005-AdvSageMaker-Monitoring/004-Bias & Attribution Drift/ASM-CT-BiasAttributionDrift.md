# Bias & Attribution Drift

**Estimated Time:** 10 Minutes

## Introduction

Data drift and model quality degradation are operational concerns. Bias drift and feature attribution drift are *ethical and regulatory* concerns. A model that was fair at deployment can become biased over time if the data or the model's learned patterns shift in ways that disproportionately affect protected groups. Similarly, a model that relied on legitimate features at training time might start depending on proxy features that correlate with sensitive attributes.

SageMaker **Clarify** integrates with Model Monitor to detect these changes. Bias monitoring tracks fairness metrics across protected groups (e.g., gender, age, ethnicity). Feature attribution monitoring tracks SHAP values to detect whether the features driving predictions have changed from the baseline. Both are deployed as monitoring schedules alongside Data Quality and Model Quality monitoring.

## Core Concepts

### Bias drift monitoring

Bias drift monitoring compares fairness metrics computed at deployment time against metrics computed on live production data. The workflow:

1. **Create a bias baseline** using SageMaker Clarify. Run a Clarify processing job on the training data, specifying:
   - **Facets:** The protected attributes to monitor (e.g., `gender`, `age_group`).
   - **Label:** The target column.
   - **Bias metrics:** Which fairness metrics to compute (see table below).
2. **Create a bias monitoring schedule** that runs Clarify on captured production data at regular intervals, comparing against the baseline.
3. **Violation reports** flag when any bias metric has changed significantly.

### Key bias metrics

| Metric | What It Measures | Formula Intuition |
| :--- | :--- | :--- |
| **Class Imbalance (CI)** | Whether protected groups are proportionally represented | Difference in group sizes |
| **Demographic Parity (DPL)** | Whether positive prediction rates are equal across groups | P(positive \| group A) - P(positive \| group B) |
| **Conditional Demographic Disparity (CDDL)** | Demographic parity conditioned on legitimate factors | Controls for confounders |
| **Treatment Equality (TE)** | Whether error rates (FP/FN) are balanced across groups | Ratio of false positives to false negatives per group |

A bias violation occurs when a metric moves beyond a threshold you define (e.g., DPL changes by more than 0.05 from baseline). The threshold depends on your regulatory context and business tolerance.

### Feature attribution drift monitoring

Feature attribution drift monitors whether the features driving predictions have changed. The workflow:

1. **Create a feature attribution baseline** using SageMaker Clarify. Run a Clarify job that computes SHAP values for the training data. The baseline records the global SHAP feature importance ranking and values.
2. **Create a feature attribution monitoring schedule** that computes SHAP values on production inference data and compares against the baseline.
3. **Violation reports** flag when feature importance has shifted significantly.

### Why feature attribution drift matters

Consider a loan approval model that was trained primarily on legitimate features (income, employment length, credit history). Over time, if the model starts relying heavily on zip code -- which correlates with race -- the model may become discriminatory even if its overall accuracy remains high. Feature attribution monitoring catches this shift by detecting that SHAP values for `zip_code` have increased significantly from the baseline.

### Configuring Clarify monitoring in the console

1. Navigate to **SageMaker > Inference > Model monitoring > Create monitoring schedule**.
2. **Monitoring type:** Select **Bias drift** or **Feature attribution drift**.
3. **Clarify configuration:**
   - **Model name:** The deployed model.
   - **Headers:** Feature names (if not embedded in the data).
   - **Facets (bias only):** The protected attribute column(s) and their sensitive values.
   - **SHAP config (attribution only):** Number of SHAP samples (controls accuracy vs. compute time).
4. **Baseline:** S3 path to the Clarify baseline output.
5. **Schedule, instance type, output path:** Same as other monitoring types.

### Combining all four monitoring types

In a production deployment, you typically run all four monitoring types on the same endpoint:

| Monitoring Type | Detects | Baseline Source | Frequency |
| :--- | :--- | :--- | :--- |
| Data Quality | Input distribution drift | Training data statistics | Hourly or daily |
| Model Quality | Prediction accuracy degradation | Evaluation set metrics | Weekly (depends on label availability) |
| Bias Drift | Fairness metric changes | Clarify bias report on training data | Weekly or monthly |
| Feature Attribution | Feature importance changes | Clarify SHAP report on training data | Weekly or monthly |

All four schedules can run independently on the same endpoint's captured data. Their violation reports and CloudWatch metrics are separate, allowing targeted alerting per monitoring type.

### Regulatory context

For models in regulated industries (financial services, healthcare, insurance), bias monitoring is not optional. Regulations like the EU AI Act and the US Fair Lending laws require ongoing monitoring of ML models for discriminatory impact. SageMaker Clarify's monitoring capabilities provide the evidence trail: baseline bias metrics, ongoing monitoring reports, and violation alerts that demonstrate compliance.

## Connecting to Practice

Bias and attribution monitoring complete the four-pillar monitoring framework. The next topic, *Drift Statistical Tests*, goes deeper into the statistical methods Model Monitor uses to quantify drift. The module assignment will require you to configure a bias monitoring schedule with a protected attribute and demonstrate violation detection when bias is injected into production data.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Clarify Bias Monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-bias-drift.html)** - *Docs*: Complete reference for bias drift monitoring configuration.
- **[Feature Attribution Monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html)** - *Docs*: Reference for SHAP-based attribution drift monitoring.

**Interactive practice**

- **[Clarify Monitoring Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_model_monitor/fairness_and_explainability)** - *Interactive*: Sample notebook demonstrating bias and feature attribution monitoring end to end.
