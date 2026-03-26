# Serverless Inference

**Estimated Time:** 10 Minutes

## Introduction

In the foundational SageMaker skill, you deployed models to **real-time endpoints** backed by always-on instances (`ml.m5.xlarge`). That architecture works well for steady, predictable traffic. But many production models receive sporadic requests -- a fraud scoring model that handles bursts during business hours and nothing overnight, or an internal tool that gets a few dozen requests per day. Keeping an `ml.m5.xlarge` running 24/7 for sporadic traffic wastes money.

SageMaker **Serverless Inference** eliminates idle costs by automatically scaling compute to zero when there are no requests and spinning up capacity on demand. You pay only for the compute time used during inference, measured in milliseconds. This topic covers when Serverless Inference is appropriate, how to configure it, and the trade-offs compared to real-time endpoints.

## Core Concepts

### How Serverless Inference works

Instead of provisioning dedicated instances, you configure a Serverless endpoint with two parameters:

- **Memory size:** The amount of RAM allocated to each inference container (1024 MB to 6144 MB, in 1024 MB increments).
- **Max concurrency:** The maximum number of concurrent inference requests the endpoint can handle (1 to 200).

SageMaker manages the underlying compute. When a request arrives, SageMaker provisions a container with your model, runs inference, returns the response, and deallocates the container after an idle timeout. If multiple requests arrive simultaneously, SageMaker provisions additional containers up to the max concurrency limit.

### Creating a Serverless endpoint in the console

1. Navigate to **SageMaker > Inference > Models** and create a model (or use an existing registered model).
2. Navigate to **SageMaker > Inference > Endpoint configurations > Create endpoint configuration**.
3. Under **Production variants**, click **Add variant**.
4. Select your model.
5. Under **Inference type**, select **Serverless**.
6. Set **Memory size** (e.g., 2048 MB for a scikit-learn model) and **Max concurrency** (e.g., 5).
7. Click **Create endpoint configuration**.
8. Navigate to **SageMaker > Inference > Endpoints > Create endpoint** and select the configuration you just created.
9. Click **Create endpoint**.

The endpoint status will show **InService** when ready. Unlike real-time endpoints, the endpoint does not provision instances at creation -- they are created on demand.

### Cold start latency

The primary trade-off of Serverless Inference is **cold start**. When the first request arrives after a period of inactivity, SageMaker must:

1. Provision a compute instance.
2. Download the model artifact from S3.
3. Load the model into memory.
4. Run inference.

This cold start adds several seconds (typically 5-30 seconds depending on model size) to the first response. Subsequent requests reuse the warm container and respond in milliseconds, similar to real-time endpoints.

**Mitigation strategies:**
- **Provisioned Concurrency (advanced):** Keep a minimum number of containers warm at all times. This adds base cost but eliminates cold starts.
- **Model size optimization:** Smaller model artifacts download faster. Compress models or use simpler algorithms.
- **Warm-up requests:** Send periodic dummy requests to keep the container warm during expected traffic hours.

### When to use Serverless Inference

| Scenario | Recommended | Reason |
| :--- | :--- | :--- |
| Traffic < 100 requests/day | Serverless | Cost savings outweigh cold start latency |
| Internal tools, batch-triggered scoring | Serverless | Sporadic, unpredictable traffic |
| Real-time user-facing API (< 100ms SLA) | Real-time endpoint | Cold start violates latency requirements |
| Steady high-volume traffic | Real-time endpoint | Always-on instances are more cost-effective |
| Traffic spikes but with latency tolerance | Serverless | Auto-scaling handles bursts, users tolerate cold starts |

### Cost comparison

- **Real-time endpoint:** Billed per instance-hour while the endpoint is InService, regardless of traffic. An `ml.m5.xlarge` running 24/7 costs approximately $170/month.
- **Serverless endpoint:** Billed per millisecond of compute time plus data processed. A model handling 1,000 requests/day at 200ms each costs approximately $3-5/month.

For low-traffic models, Serverless Inference can reduce costs by 95% or more compared to real-time endpoints.

### Limitations

- **Max payload:** 6 MB request payload, 6 MB response payload.
- **Max response time:** 60 seconds per inference call.
- **No GPU:** Serverless Inference runs on CPU-only infrastructure.
- **No multi-model endpoints:** Each Serverless endpoint hosts a single model.
- **Container image constraints:** The model container must be compatible with SageMaker's Serverless runtime.

## Connecting to Practice

Serverless Inference is the first of several inference patterns you will learn in this module. The next topic, *Asynchronous Inference*, covers an alternative for large-payload, long-running predictions. *Inference Decision Matrix* will synthesize all patterns into a decision framework. The module assignment will require you to deploy a model as a Serverless endpoint and measure cold start latency vs. warm latency.

## Further Learning & Resources

**Documentation and reading**

- **[Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)** - *Docs*: Complete configuration reference for Serverless endpoints.
- **[Serverless Inference Pricing](https://aws.amazon.com/sagemaker/pricing/)** - *Docs*: Detailed pricing model for Serverless compute time and data processing.

**Interactive practice**

- **[Deploy a Serverless Endpoint](https://github.com/aws/amazon-sagemaker-examples/tree/main/serverless-inference)** - *Interactive*: Sample notebook demonstrating Serverless endpoint creation and invocation.
