# Auto-scaling Policies

**Estimated Time:** 10 Minutes

## Introduction

A real-time endpoint with a fixed number of instances either wastes money during low traffic (over-provisioned) or drops requests during high traffic (under-provisioned). SageMaker endpoint auto-scaling dynamically adjusts the instance count based on traffic patterns, maintaining performance during peaks and reducing costs during troughs.

This reading covers how to configure auto-scaling policies for SageMaker endpoints, the types of scaling policies available, and best practices for setting scaling targets and cooldowns.

## Core Concepts

### Auto-scaling components

SageMaker uses **Application Auto Scaling** (the same service that scales DynamoDB, ECS, and other AWS resources) to manage endpoint instance counts. The configuration has three parts:

1. **Scalable target:** Registers the endpoint variant as a scalable resource and defines the minimum and maximum instance counts.
2. **Scaling policy:** Defines *when* and *how much* to scale based on metrics.
3. **Cooldown periods:** Define how long to wait after a scaling event before another scaling event can occur.

### Configuring auto-scaling in the console

1. Navigate to **SageMaker > Inference > Endpoints > [your endpoint]**.
2. Click the **Endpoint runtime settings** tab.
3. Under the production variant, click **Configure auto-scaling**.
4. **Minimum instances:** The floor (e.g., 1). The endpoint will never scale below this.
5. **Maximum instances:** The ceiling (e.g., 10). The endpoint will never scale above this.
6. **Scaling policy:** Choose a policy type (see below).
7. Click **Save**.

### Target tracking policy

The most common and recommended policy type. You specify a **target metric value**, and SageMaker automatically adjusts instance count to maintain that target.

**Configuration:**
- **Metric:** `SageMakerVariantInvocationsPerInstance` (the average number of invocations per instance per minute).
- **Target value:** The desired invocations per instance per minute (e.g., 70).
- **Scale-in cooldown:** Seconds to wait before removing instances after a scale-in (e.g., 300 seconds).
- **Scale-out cooldown:** Seconds to wait before adding instances after a scale-out (e.g., 60 seconds).

When actual invocations per instance exceed the target, SageMaker adds instances. When they drop below, SageMaker removes instances (after the cooldown).

### Step scaling policy

For more granular control, step scaling defines specific scaling actions based on metric thresholds:

- When `InvocationsPerInstance` > 100 for 3 minutes: add 2 instances.
- When `InvocationsPerInstance` > 200 for 1 minute: add 5 instances.
- When `InvocationsPerInstance` < 20 for 10 minutes: remove 1 instance.

Step scaling is useful when you need different scaling speeds for different traffic levels (e.g., aggressive scale-out during sudden spikes, conservative scale-in during gradual declines).

### Custom metrics for scaling

Beyond the default `InvocationsPerInstance` metric, you can scale on custom CloudWatch metrics:

- **`ModelLatency`:** Average inference latency. Scale out when latency exceeds an SLA threshold (e.g., 200ms).
- **Queue depth (Async endpoints):** Scale based on the number of queued requests in the `ApproximateBacklogSizePerInstance` metric.
- **Custom application metrics:** If your inference container emits custom CloudWatch metrics, you can use them as scaling triggers.

### Cooldown best practices

Cooldowns prevent "flapping" -- rapid scale-out/scale-in cycles caused by metric oscillation around the target value.

- **Scale-out cooldown:** Keep short (60-120 seconds). You want to respond quickly to traffic increases to maintain latency SLAs.
- **Scale-in cooldown:** Keep longer (300-600 seconds). Removing instances too quickly during a temporary traffic lull risks under-provisioning when traffic returns. In-flight requests on the removed instance must complete before termination.

### Auto-scaling with Serverless and Async endpoints

- **Serverless endpoints:** Scaling is fully managed. You only set `MaxConcurrency`. SageMaker handles container provisioning automatically.
- **Async endpoints:** Support auto-scaling including scale-to-zero. The scaling metric is `ApproximateBacklogSizePerInstance` (how many queued requests per instance). Scale to zero when the backlog is empty.

### Monitoring auto-scaling

View scaling activity in the console:

1. **SageMaker > Endpoints > [your endpoint] > CloudWatch metrics:** View `InvocationsPerInstance`, `OverheadLatency`, and instance count over time.
2. **Application Auto Scaling > Scaling activities:** View the scaling event history -- when instances were added or removed, and which metric triggered the action.
3. **CloudWatch Dashboards:** Create a custom dashboard combining invocation count, latency, and instance count to visualize scaling behavior in context.

## Connecting to Practice

Auto-scaling keeps endpoints responsive without manual intervention. The next topic, *Inference Recommender*, covers SageMaker's tool for data-driven instance type and configuration selection. The module assignment will require you to configure a target tracking auto-scaling policy, generate a load test, and verify that the endpoint scales correctly.

## Further Learning & Resources

**Documentation and reading**

- **[Auto-scaling SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)** - *Docs*: Complete reference for auto-scaling configuration, policies, and metrics.
- **[Application Auto Scaling](https://docs.aws.amazon.com/autoscaling/application/userguide/what-is-application-auto-scaling.html)** - *Docs*: General documentation for the auto-scaling service SageMaker uses.

**Interactive practice**

- **[Endpoint Auto-scaling Lab](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/autoscaling)** - *Interactive*: Sample notebook demonstrating auto-scaling policy configuration and load testing.
