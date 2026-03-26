# Invoking Endpoints

**Estimated Time:** 10 Minutes

## Introduction

In the *Real-time Inference Endpoints* reading you deployed a model and saw it reach **InService** status. The endpoint is live, the container is running, and the model is loaded in memory. But an endpoint that nobody talks to is just an expensive idle server. The final step is **invocation**: sending data to the endpoint and receiving predictions back.

In the AIML Foundations module, invoking a trained model was one line of code:

```python
predictions = model.predict(X_test)
```

The model was in memory on your laptop. The data was a NumPy array already in the right shape. There was no network, no serialization, no content types to worry about. With a SageMaker endpoint, the model lives on a remote server. Your data has to travel over HTTPS, which means it has to be serialized into a format the endpoint understands (CSV or JSON), sent as an HTTP request body, and deserialized on the other side. The response comes back the same way.

This reading covers how to invoke endpoints using `boto3` (the low-level AWS SDK) and the SageMaker `Predictor` class (the high-level SDK), how to handle content types and serialization, and how to debug common invocation errors.

## Core Concepts

### The invocation flow

When you invoke a SageMaker endpoint, the following happens:

1. Your client code serializes the input data (a DataFrame row, a JSON object, a CSV string) into a request body.
2. The client sends an HTTPS POST request to the SageMaker runtime API, specifying the endpoint name and content type.
3. SageMaker routes the request to the inference container running on the endpoint instance.
4. The container deserializes the request body, runs it through the model, and serializes the prediction into a response body.
5. SageMaker returns the response to your client.
6. Your client deserializes the response into a usable format (a number, a class label, a probability distribution).

The entire round trip typically takes 50-200 milliseconds for a simple tabular model on `ml.m5.xlarge`.

### Testing from a Studio notebook

The most direct way to test an endpoint is from a SageMaker Studio notebook. Since Studio runs inside your AWS account, it has network access to your endpoints and inherits your User Profile's execution role for permissions.

1. **Open SageMaker Studio** from the console (as you learned in the *Studio Domains & Profiles* reading).
2. **Create or open a notebook.** Choose a Python 3 kernel.
3. **Write invocation code** using either `boto3` or the SageMaker `Predictor` class (both covered below).
4. **Run the cell.** The response appears in the notebook output.

This is the recommended workflow for testing during development. You do not need to leave the browser -- Studio provides the notebook environment and the AWS credentials in one place.

### Invoking with boto3

`boto3` is the general-purpose AWS SDK for Python. The `sagemaker-runtime` client provides the `invoke_endpoint` method, which is the lowest-level way to call an endpoint.

```python
import boto3
import json

runtime = boto3.client("sagemaker-runtime")

response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v3-endpoint",
    ContentType="text/csv",
    Body="0.5,1200,3,0,1\n",
)

result = response["Body"].read().decode("utf-8")
print(result)
```

Let us break down each parameter:

**`EndpointName`** is the name you gave the endpoint when you created it in the console. This is how SageMaker routes your request to the correct model.

**`ContentType`** tells the inference container how to parse the request body. Common values:
- `"text/csv"` -- the body is a CSV-formatted string. Each row is one sample; columns are features separated by commas.
- `"application/json"` -- the body is a JSON string. The expected structure depends on the model's inference script.

**`Body`** is the raw data payload as a string or bytes. For CSV, this is a comma-separated row (or multiple rows separated by newlines). For JSON, this is a JSON-encoded string.

**The response** is a dictionary. `response["Body"]` is a streaming object; you call `.read().decode("utf-8")` to get the prediction as a string. For a classification model, this might be `"1"` (fraud) or `"0"` (not fraud). For a regression model, it might be `"42.5"`.

### Content type and serialization

The content type must match what the inference container expects. For scikit-learn Script Mode models, the default inference handler accepts both CSV and JSON:

**CSV format:**

```python
Body = "0.5,1200,3,0,1\n"
ContentType = "text/csv"
```

Multiple samples can be sent in one request, each on a new line:

```python
Body = "0.5,1200,3,0,1\n0.8,500,1,1,0\n"
```

**JSON format:**

```python
import json

payload = {"instances": [[0.5, 1200, 3, 0, 1]]}
Body = json.dumps(payload)
ContentType = "application/json"
```

CSV is simpler for tabular data. JSON is more flexible and supports nested structures, which is useful for NLP or image models. Choose based on your model's input requirements.

### Invoking with the SageMaker Predictor class

The SageMaker SDK provides a higher-level `Predictor` class that handles serialization and deserialization automatically:

```python
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = Predictor(
    endpoint_name="fraud-rf-v3-endpoint",
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
)

result = predictor.predict([0.5, 1200, 3, 0, 1])
print(result)
```

The `Predictor` class wraps `invoke_endpoint` and adds two conveniences:

**Serializers** convert your Python data (lists, arrays, DataFrames) into the wire format. `CSVSerializer()` converts a list to a CSV string. `JSONSerializer()` converts a dictionary to JSON. You do not need to manually format the `Body` parameter.

**Deserializers** convert the response body back into a Python object. `JSONDeserializer()` parses JSON into a dictionary. `CSVDeserializer()` parses CSV into a list of strings.

If you called `estimator.deploy()` in the previous reading, it returns a `Predictor` object preconfigured with the correct endpoint name and default serializers:

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
)
result = predictor.predict([0.5, 1200, 3, 0, 1])
```

### From model.predict() to invoke_endpoint

Here is the conceptual mapping between local inference and SageMaker inference:

| Local (AIML Foundations) | SageMaker Endpoint |
| :--- | :--- |
| `model = joblib.load("model.pkl")` | Model loaded automatically by the inference container |
| `X_test` as a NumPy array | Serialized as CSV or JSON in the request body |
| `model.predict(X_test)` | `runtime.invoke_endpoint(...)` or `predictor.predict(...)` |
| Return value as a NumPy array | Response body as a string, deserialized by your client |

The logic is identical: data goes in, predictions come out. The difference is the network layer in between, which requires serialization on the way in and deserialization on the way out.

### Handling errors

Common invocation errors and how to debug them:

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `ValidationError: Endpoint not found` | Wrong endpoint name or endpoint was deleted | Check the name in **Inference > Endpoints** in the console |
| `ModelError` (HTTP 400/500) | The inference container could not process your input | Check the content type, data format, and CloudWatch logs for the endpoint |
| `AccessDeniedException` | Your execution role or IAM user lacks `sagemaker:InvokeEndpoint` permission | Add `sagemaker:InvokeEndpoint` to the role's policy in the IAM console |
| Timeout | The model takes too long to respond | Check instance type sizing; consider async inference for heavy models |

When you get a `ModelError`, the most useful debugging step is checking CloudWatch Logs. Endpoint logs live in the log group `/aws/sagemaker/Endpoints/<endpoint-name>`. The error traceback from the inference container appears there and usually points directly to the issue (wrong input shape, missing feature columns, deserialization failure).

### Cleanup reminder

Endpoints incur charges every second they are running. After testing your invocations, delete the endpoint and associated resources:

```python
predictor.delete_endpoint()
predictor.delete_model()
```

Or use the console: **Inference > Endpoints > Select > Actions > Delete**. Then delete the Endpoint Configuration and Model as described in the *Real-time Inference Endpoints* reading.

Verify in the console that no endpoints show **InService** status. Check the **Billing & Cost Management** console to confirm.

## Connecting to Practice

This reading completes the deployment workflow: you trained a model, registered it, approved it, deployed it to an endpoint, and invoked it for predictions. In the *Invoking Endpoints Video*, you will see a live demonstration of `boto3` and `Predictor` invocations from a Studio notebook. In the module lecture, you will build and test the full pipeline. And in the module assignment, you will invoke endpoints programmatically and handle the results.

The most useful thing you can do right now is open a Studio notebook (or any Python environment with `boto3` configured) and write a simple `invoke_endpoint` call against an active endpoint. Send a single row of data, read the response, and print the prediction. Then delete the endpoint. That complete loop -- invoke, read, clean up -- is the pattern you will use in every deployment exercise.

---

## Further Learning & Resources

**Documentation and reading**

- **[InvokeEndpoint API Reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html)** - *Docs*: The complete API reference for `invoke_endpoint`, including all parameters, content types, and response format.
- **[SageMaker Python SDK - Predictors](https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html)** - *Docs*: The Predictor class reference with all serializer/deserializer options and configuration parameters.

**Interactive practice**

- **[SageMaker Examples - Inference](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk)** - *Interactive*: Runnable notebooks demonstrating endpoint invocation with various frameworks and data formats.
- **[AWS Hands-On: Deploy and Invoke](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-deploy-model-to-real-time-inference-endpoint/)** - *Interactive*: A free guided lab covering the complete deploy-invoke-cleanup cycle in your own console.
