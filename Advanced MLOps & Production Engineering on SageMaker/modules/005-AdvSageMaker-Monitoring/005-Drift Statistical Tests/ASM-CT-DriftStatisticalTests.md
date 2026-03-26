# Drift Statistical Tests

**Estimated Time:** 10 Minutes

## Introduction

When Model Monitor reports that a feature has "drifted," how does it quantify the drift? The answer is statistical hypothesis testing: Model Monitor applies specific tests to compare the distribution of live data against the baseline distribution. Understanding which tests are used, how they work, and what their results mean is essential for correctly interpreting monitoring reports and setting appropriate thresholds.

This reading covers the two primary statistical tests used by SageMaker Model Monitor for data quality drift detection: the **Kolmogorov-Smirnov (K-S) test** for continuous features and the **Chi-Squared test** for categorical features.

## Core Concepts

### Kolmogorov-Smirnov (K-S) test

The K-S test is used for **continuous (numeric) features**. It compares the cumulative distribution functions (CDFs) of the baseline and live data.

**How it works:**
1. Compute the empirical CDF of the baseline data (F_baseline).
2. Compute the empirical CDF of the live data (F_live).
3. The K-S statistic (D) is the maximum absolute difference between the two CDFs: D = max|F_baseline(x) - F_live(x)|.
4. Compare D against a critical value (based on sample sizes and significance level) to determine if the difference is statistically significant.

**Interpreting results:**
- **D close to 0:** Distributions are similar. No drift detected.
- **D close to 1:** Distributions are completely different. Severe drift.
- **p-value < significance level (e.g., 0.05):** Reject the null hypothesis (no drift). The distributions are statistically different.

**In Model Monitor reports:** The K-S statistic and p-value appear in the violation details for numeric features flagged with `baseline_drift_check`.

### Chi-Squared test

The Chi-Squared test is used for **categorical features**. It compares the frequency distribution of categories in the baseline against the live data.

**How it works:**
1. Count the frequency of each category in the baseline data (expected frequencies).
2. Count the frequency of each category in the live data (observed frequencies).
3. Compute the Chi-Squared statistic: the sum of (observed - expected)^2 / expected across all categories.
4. Compare the statistic against the Chi-Squared distribution with (k-1) degrees of freedom (where k is the number of categories).

**Interpreting results:**
- **Chi-Squared close to 0:** Category proportions match the baseline. No drift.
- **Large Chi-Squared:** Category proportions have changed significantly.
- **p-value < significance level:** The distribution of categories has shifted.

**In Model Monitor reports:** The Chi-Squared statistic and p-value appear in the violation details for categorical features flagged with `baseline_drift_check`.

### Configuring drift thresholds

Model Monitor allows you to configure the sensitivity of drift detection by editing the `constraints.json` file:

- **`monitoring_config.evaluate_constraints`:** Enable or disable constraint evaluation.
- **`monitoring_config.distribution_constraints.comparison_threshold`:** The maximum acceptable D statistic (K-S) or Chi-Squared p-value for each feature. Lower thresholds make monitoring more sensitive (more violations), higher thresholds make it more tolerant.

Default behavior: Model Monitor uses a p-value threshold of 0.05. You can adjust this per feature:

- For business-critical features (e.g., `transaction_amount`), use a stricter threshold (0.01) to catch small shifts early.
- For noisy features (e.g., `user_agent_string`), use a relaxed threshold (0.10) to avoid false alarms.

### When drift is detected but model quality is fine

A common scenario: Data Quality Monitoring flags drift, but Model Quality Monitoring shows no degradation. This happens when:

- The drifted feature has low importance (the model does not rely on it heavily).
- The drift is within the model's generalization capacity (the model handles slightly different distributions without degrading).

In these cases, the drift violation is an informational signal, not an emergency. Check SHAP feature importance: if the drifted feature is in the bottom 20% of importance, it is likely a false alarm. If it is in the top 5, it warrants investigation even if model quality has not degraded yet (degradation may be imminent).

### Limitations of statistical tests

- **Sample size sensitivity:** With very large samples (millions of predictions), even trivially small differences become statistically significant. A K-S p-value of 0.001 on a million-row dataset might correspond to a mean shift of 0.1%, which is operationally irrelevant. Supplement p-values with effect size (the D statistic itself).
- **Multivariate drift:** K-S and Chi-Squared test each feature independently. They do not detect drift in the *joint* distribution of features (correlations between features changing). Detecting multivariate drift requires custom monitoring logic.
- **Temporal patterns:** These tests compare aggregate distributions. They do not detect temporal patterns (e.g., drift that occurs only on weekends).

### Custom statistical tests

For monitoring needs beyond K-S and Chi-Squared, you can implement custom monitoring logic:

1. Write a custom Processing Job script that reads captured data and computes your preferred drift metric (e.g., Population Stability Index, Jensen-Shannon divergence, Wasserstein distance).
2. Schedule this script as a Processing Job on a recurring basis.
3. Emit custom CloudWatch metrics for alerting.

This approach is common in financial services, where regulators require specific drift metrics that differ from Model Monitor's defaults.

## Connecting to Practice

Understanding the statistical tests behind Model Monitor reports enables you to set appropriate thresholds and avoid both false alarms and missed drift. The next topic, *Monitor & Pipeline Automation*, shows how to integrate monitoring into automated retraining Pipelines. The module assignment will require you to analyze a monitoring report, identify the statistical test used for each violated feature, and justify the threshold settings.

## Further Learning & Resources

**Documentation and reading**

- **[Model Monitor Constraint Violations](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-violations.html)** - *Docs*: Guide to interpreting violation types, statistics, and thresholds.
- **[K-S Test (Wikipedia)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)** - *Reference*: Mathematical description and properties of the K-S test.
- **[Chi-Squared Test (Wikipedia)](https://en.wikipedia.org/wiki/Chi-squared_test)** - *Reference*: Mathematical description and applications of the Chi-Squared test.

**Interactive practice**

- **[Custom Monitoring Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_model_monitor/custom_monitoring)** - *Interactive*: Sample notebook demonstrating custom monitoring with user-defined statistical tests.
