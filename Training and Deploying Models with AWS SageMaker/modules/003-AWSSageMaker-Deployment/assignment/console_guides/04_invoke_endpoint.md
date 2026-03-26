# Guide 4: Invoke the Endpoint

Your endpoint is live and waiting for prediction requests. In this guide, you will send requests from a Studio notebook using the `boto3` runtime client, observe the responses, and try intentional errors to understand error handling.

---

## Steps

### Step 1 -- Open a Studio Notebook

1. From the **SageMaker console**, go to **Domains** -> **fraudshield-domain**.
2. Click **Open Studio** next to your default user profile.
3. Open a notebook (you can reuse the one from Module 2 or create a new one).

### Step 2 -- Invoke with boto3 (CSV Format)

Run this cell to send a prediction request:

```python
import boto3
import json

# Create the SageMaker runtime client
runtime = boto3.client("sagemaker-runtime")

# Prepare a CSV payload (one row of features)
# Columns: amount, hour_of_day, is_international, merchant_category,
#           customer_tenure_days, num_transactions_24h
csv_payload = "2500.00,3,1,3,45,8"  # a suspicious-looking transaction

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v1-endpoint",
    ContentType="text/csv",
    Body=csv_payload,
)

# Read the response
result = response["Body"].read().decode("utf-8")
print(f"Prediction: {result}")
print(f"HTTP Status: {response['ResponseMetadata']['HTTPStatusCode']}")
```

Observe:
- The **prediction result** (a class label or probability depending on your model)
- The **HTTP status code** (200 = success)

### Step 3 -- Invoke with Multiple Rows

Send multiple predictions in a single request:

```python
# Multiple rows, one per line
# Columns: amount, hour_of_day, is_international, merchant_category,
#           customer_tenure_days, num_transactions_24h
multi_row_payload = """2500.00,3,1,3,45,8
55.00,14,0,0,1200,2
8900.00,1,1,2,30,12"""

response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v1-endpoint",
    ContentType="text/csv",
    Body=multi_row_payload,
)

result = response["Body"].read().decode("utf-8")
print(f"Predictions:\n{result}")
```

The response should contain one prediction per input row.

### Step 4 -- Invoke with JSON Format (Optional)

If your model supports JSON input:

```python
json_payload = json.dumps({
    "instances": [[2500.00, 3, 1, 3, 45, 8]]
})

response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v1-endpoint",
    ContentType="application/json",
    Body=json_payload,
)

result = response["Body"].read().decode("utf-8")
print(f"JSON Prediction: {result}")
```

Note: Whether JSON works depends on how the inference container deserializes input. The scikit-learn container natively supports CSV; JSON support depends on the container version and configuration.

### Step 5 -- Try the SageMaker Predictor (Alternative)

The SageMaker SDK provides a higher-level `Predictor` class:

```python
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

predictor = Predictor(
    endpoint_name="fraud-rf-v1-endpoint",
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer(),
)

result = predictor.predict("2500.00,3,1,3,45,8")
print(f"Predictor result: {result}")
```

The `Predictor` handles serialization/deserialization automatically, but the `boto3` runtime client gives you more control over the request.

### Step 6 -- Trigger Intentional Errors

Try sending bad input to see how errors appear:

**Wrong feature count:**
```python
# Too few features
try:
    response = runtime.invoke_endpoint(
        EndpointName="fraud-rf-v1-endpoint",
        ContentType="text/csv",
        Body="2500.00,3",  # only 2 of 6 features
    )
    result = response["Body"].read().decode("utf-8")
    print(result)
except Exception as e:
    print(f"Error: {e}")
```

**Wrong endpoint name:**
```python
try:
    response = runtime.invoke_endpoint(
        EndpointName="nonexistent-endpoint",
        ContentType="text/csv",
        Body="2500.00,3,1,3,45,8",
    )
except Exception as e:
    print(f"Error: {e}")
```

### Step 7 -- Check CloudWatch Logs for the Endpoint

1. Go to the **CloudWatch console** (search for "CloudWatch" in the top search bar).
2. Click **Log groups** in the left navigation.
3. Search for `/aws/sagemaker/Endpoints/fraud-rf-v1-endpoint`.
4. Click on the log group and then on the most recent log stream.
5. Look for:
   - Successful prediction log entries
   - Error tracebacks from the intentional errors you triggered
   - Container startup messages

These logs are your primary debugging tool for production inference issues.

---

## Presentation Checkpoint

Be prepared to show:
- A successful `invoke_endpoint` call and the prediction result
- The multi-row prediction response
- The error message when sending wrong input (wrong feature count or wrong endpoint name)
- The CloudWatch Logs for the endpoint
- Explain: What are the four required parameters for `invoke_endpoint`? (EndpointName, ContentType, Body, and optionally Accept)
- Explain: What is the difference between using `boto3` runtime client vs. the SageMaker `Predictor` class? (boto3 gives low-level control over the HTTP request; Predictor handles serialization/deserialization automatically and is more convenient for notebook work)
- Explain: Where do you look when an endpoint returns an error? (CloudWatch Logs for the endpoint, under `/aws/sagemaker/Endpoints/<endpoint-name>`)

---

## Key Concepts

- **`sagemaker-runtime` Client:** The boto3 client specifically for invoking SageMaker endpoints. Different from the regular `sagemaker` client used for management operations.
- **ContentType:** Tells the endpoint how to deserialize the input. Must match what the inference container expects (`text/csv` for scikit-learn containers).
- **Serialization/Deserialization:** The process of converting data between Python objects and wire formats (CSV strings, JSON strings). The `Predictor` class automates this.
- **Endpoint CloudWatch Logs:** Unlike training jobs (which are ephemeral), endpoint logs accumulate over time as the endpoint serves requests. This is essential for monitoring production models.

---

## AIML Connection

The *Evaluation Metrics* reading emphasized that model performance in production can differ from training metrics. Each prediction response is a data point -- over time, you would compare these predictions against actual outcomes to detect model drift. This is the monitoring stage of the ML lifecycle that Module 4 will address.
