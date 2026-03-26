# DeepAR Time Series

**Estimated Time:** 10 Minutes

## Introduction

Time-series forecasting -- predicting future values based on historical patterns -- is critical for demand planning, capacity management, financial projections, and operational scheduling. Traditional statistical methods (ARIMA, exponential smoothing) work well for individual time series but struggle when you have hundreds or thousands of related series that share common patterns.

SageMaker's **DeepAR** is a built-in algorithm that uses a deep learning approach (autoregressive recurrent neural networks) to forecast multiple related time series simultaneously. By training on all series together, DeepAR learns shared patterns (seasonality, trend shapes) across series while respecting each series' individual characteristics. This makes it particularly effective for scenarios like forecasting sales across thousands of products, predicting energy consumption across hundreds of meters, or estimating demand across multiple warehouse locations.

## Core Concepts

### How DeepAR differs from classical forecasting

| Feature | Classical (ARIMA/ETS) | DeepAR |
| :--- | :--- | :--- |
| Model per series | One model per time series | Single model for all series |
| Cold start | Requires history for each series | Can forecast new series with covariates |
| Feature support | Limited exogenous variables | Categorical and continuous covariates |
| Uncertainty quantification | Point forecasts (or simple intervals) | Full probabilistic forecasts (quantiles) |
| Scale | Practical for ~10s of series | Designed for 100s to 1000s of series |

### Input format

DeepAR expects data in JSON Lines format. Each line represents one time series:

```json
{"start": "2024-01-01 00:00:00", "target": [10, 15, 12, 18, 22, 19, 25], "cat": [0]}
{"start": "2024-01-01 00:00:00", "target": [5, 8, 6, 9, 11, 10, 14], "cat": [1]}
```

- **`start`:** The timestamp of the first observation.
- **`target`:** The sequence of values to forecast.
- **`cat` (optional):** Categorical features that identify the series (e.g., product category, region). DeepAR uses these to learn group-specific patterns.
- **`dynamic_feat` (optional):** Time-varying features that are known for both historical and future periods (e.g., holidays, promotions, day-of-week indicators).

### Key hyperparameters

| Parameter | Purpose | Typical values |
| :--- | :--- | :--- |
| `prediction_length` | Number of future time steps to forecast | Problem-specific (e.g., 7 for weekly, 30 for monthly) |
| `context_length` | Number of historical time steps the model sees when making a prediction | Often set equal to `prediction_length` or 2-3x |
| `num_cells` | Size of the RNN hidden state | 30-100 |
| `num_layers` | Depth of the RNN | 1-3 |
| `epochs` | Training passes | 100-500 |
| `likelihood` | Distribution family for outputs | `gaussian` (continuous), `negative-binomial` (count data) |
| `time_freq` | Frequency of the time series | `H` (hourly), `D` (daily), `W` (weekly), `M` (monthly) |

### Configuring a DeepAR training job

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. **Algorithm:** Select **DeepAR** from the built-in list.
3. **Input data:** Set the `train` channel to the S3 path of your JSON Lines file. Optionally set a `test` channel with held-out data for evaluation.
4. **Hyperparameters:** At minimum, set `prediction_length`, `time_freq`, and `context_length`.
5. **Instance type:** `ml.m5.xlarge` for CPU training (sufficient for datasets with up to ~10,000 series). For larger datasets or faster training, use `ml.p3.2xlarge` (GPU).

### Probabilistic forecasts

Unlike models that produce a single predicted value, DeepAR generates **quantile forecasts**. When you invoke the deployed endpoint, you specify which quantiles you want (e.g., 0.1, 0.5, 0.9). The response includes a predicted value for each quantile at each future time step.

- **p50 (median):** The central forecast. Use this as your "best guess."
- **p10 and p90:** Prediction interval bounds. There is approximately an 80% probability that the actual value falls between p10 and p90.

This probabilistic output is essential for business decision-making: a supply chain manager wants to know not just the expected demand but the range of possible outcomes to plan inventory buffers.

### Using categorical covariates

The `cat` field is one of DeepAR's most powerful features. By assigning a categorical identifier to each time series, DeepAR learns embeddings for each category and uses them to inform predictions. This enables:

- **Cold-start forecasting:** For a new product with no history, assign it the same category as similar existing products. DeepAR uses the learned category embedding to generate a reasonable forecast.
- **Cross-series learning:** Products in the same category share seasonality patterns. DeepAR captures these shared patterns automatically.

## Connecting to Practice

DeepAR completes the survey of SageMaker's built-in algorithms for different data types: XGBoost for tabular classification/regression, K-Means for clustering, RCF for anomaly detection, BlazingText for text, and DeepAR for time series. The remaining topics in this module cover hyperparameter tuning, which applies across all these algorithms. In the module assignment, you will train a DeepAR model on multi-series time data with categorical covariates and evaluate its quantile forecasts.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker DeepAR Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)** - *Docs*: Complete reference for input format, hyperparameters, and inference output.
- **[DeepAR Paper (Salinas et al., 2020)](https://doi.org/10.1016/j.ijforecast.2019.07.001)** - *Reading*: The research paper describing the DeepAR architecture and its evaluation against classical methods.

**Interactive practice**

- **[Time Series Forecasting with DeepAR](https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms/deepar_electricity)** - *Interactive*: Sample notebook demonstrating DeepAR on electricity consumption data with visualization of quantile forecasts.
