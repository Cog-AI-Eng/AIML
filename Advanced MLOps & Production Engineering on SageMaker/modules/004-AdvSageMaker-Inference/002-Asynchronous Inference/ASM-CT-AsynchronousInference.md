# Asynchronous Inference

**Estimated Time:** 10 Minutes

## Introduction

Real-time inference endpoints respond synchronously: the client sends a request and waits for the prediction. This works when inference takes milliseconds, but some models require minutes to process -- large document analysis, high-resolution image processing, or complex ensemble predictions that chain multiple models. For these workloads, the client cannot wait for a synchronous response, and a 60-second API timeout is insufficient.

SageMaker **Asynchronous Inference** decouples the request from the response. The client submits a request (an input payload stored in S3), SageMaker queues it, processes it when capacity is available, and writes the result to a specified S3 output location. The client can poll S3 for the result or receive a notification via Amazon SNS when processing completes.

## Core Concepts

### How Asynchronous Inference works

1. **Client uploads input** to S3 (e.g., `s3://bucket/input/request-001.csv`).
2. **Client invokes the endpoint** with the `InvokeEndpointAsync` API, passing the S3 input URI.
3. **SageMaker queues the request** and returns immediately with a response token and output location.
4. **SageMaker processes the request** asynchronously. If no instances are running, it scales up from zero (like Serverless, Async endpoints support scale-to-zero).
5. **SageMaker writes the result** to S3 (e.g., `s3://bucket/output/request-001.out`).
6. **SageMaker sends an SNS notification** (if configured) indicating success or failure.

### Creating an Async endpoint in the console

1. Navigate to **SageMaker > Inference > Endpoint configurations > Create endpoint configuration**.
2. Under **Production variants**, add a variant and select your model.
3. Under **Async inference configuration**, enable async inference.
4. **S3 output path:** Specify the S3 prefix where results will be written (e.g., `s3://bucket/async-output/`).
5. **SNS topics (optional):** Specify an SNS topic ARN for success notifications and another for error notifications.
6. **Max concurrent invocations per instance:** Controls how many requests each instance processes simultaneously. Lower values reduce per-request latency; higher values improve throughput.
7. Set the instance type (e.g., `ml.m5.xlarge`) and min/max instance count.
8. Create the endpoint configuration and then create an endpoint using it.

### Scale-to-zero

Async endpoints support scaling the instance count to zero during idle periods, similar to Serverless Inference:

- Set the **minimum instance count** to 0 in the auto-scaling configuration.
- When no requests are queued, SageMaker removes all instances.
- When a new request arrives, SageMaker provisions an instance and processes it.

The trade-off is the same cold start latency as Serverless: the first request after an idle period takes longer. But unlike Serverless, Async endpoints support any instance type (including GPU) and arbitrarily large payloads.

### Notification workflow

The SNS integration enables event-driven architectures:

1. Configure success and error SNS topics in the endpoint configuration.
2. Subscribe a Lambda function, SQS queue, or email to the SNS topics.
3. When inference completes, the Lambda function can read the result from S3, post-process it (e.g., parse predictions, update a database), and trigger downstream actions.

This pattern is common for batch-like workflows where many requests are submitted at once and results are processed in aggregate after all complete.

### When to use Async vs. other inference types

| Factor | Async Inference | Real-time | Serverless | Batch Transform |
| :--- | :--- | :--- | :--- | :--- |
| Latency tolerance | Minutes acceptable | Milliseconds required | Seconds (cold start) | Hours acceptable |
| Payload size | Up to 1 GB | Up to 6 MB | Up to 6 MB | Unlimited |
| Processing time | Up to 60 minutes | Up to 60 seconds | Up to 60 seconds | No limit |
| Scale to zero | Yes | No | Yes | N/A (job-based) |
| GPU support | Yes | Yes | No | Yes |
| Ideal for | Large payloads, long processing | Low-latency APIs | Sporadic, small requests | Bulk scoring datasets |

### Cost considerations

Async endpoint billing follows the same model as real-time endpoints: you pay for instance-hours while instances are running. With scale-to-zero enabled, you only pay when there are queued requests. This makes Async Inference cost-effective for workloads with variable demand: you get GPU support and large payload handling without the 24/7 cost of a real-time endpoint.

## Connecting to Practice

Async Inference fills the gap between real-time and batch processing. The next topic, *Inference Decision Matrix*, provides a structured framework for choosing among all inference options. The module assignment will require you to deploy an Async endpoint with SNS notifications and demonstrate the end-to-end flow from request submission to S3 output retrieval.

## Further Learning & Resources

**Documentation and reading**

- **[Asynchronous Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)** - *Docs*: Complete reference for Async endpoint configuration, invocation, and monitoring.
- **[Async Inference Auto-scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html)** - *Docs*: Guide to configuring scale-to-zero and custom scaling policies.

**Interactive practice**

- **[Async Inference Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/async-inference)** - *Interactive*: Sample notebook demonstrating Async endpoint creation, invocation, and SNS notification handling.
