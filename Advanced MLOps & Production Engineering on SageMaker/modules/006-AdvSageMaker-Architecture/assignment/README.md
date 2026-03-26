# Advanced SageMaker Architecture Lab

## Scenario
FraudShield Risk Analytics is preparing for its busiest quarter. Transaction volume is expected to triple during the holiday season, and leadership wants assurance that the ML infrastructure can scale without breaking the budget. Your mission is to optimize the FraudShield platform for cost, performance, and operational visibility. Every dollar saved on training and every millisecond shaved off inference latency translates directly to business value.

In this lab you will reduce training costs with Managed Spot Instances, right-size your compute by analyzing CloudWatch metrics, configure auto-scaling so endpoints expand and contract with demand, use Inference Recommender to find the optimal instance type, and build a CloudWatch dashboard that gives the operations team a single pane of glass for the entire FraudShield inference fleet. When complete, the platform will be production-hardened for the surge ahead.

---

## Learning Objectives
By completing this lab you will demonstrate the ability to:
1. Launch a training job with Managed Spot Instances and checkpointing to reduce costs
2. Analyze CloudWatch CPU and memory metrics to right-size training instances
3. Configure auto-scaling policies on a SageMaker endpoint
4. Run an Inference Recommender job and interpret instance comparison results
5. Build a CloudWatch dashboard with endpoint performance and error widgets
6. Clean up all architecture lab resources

---

## Prerequisites
- AWS account with SageMaker, CloudWatch, and Application Auto Scaling permissions
- A trained FraudShield model artifact in S3
- A deployed real-time FraudShield endpoint in us-east-1
- An S3 bucket named `sagemaker-fraudshield-<account-id>` in us-east-1
- Familiarity with SageMaker training jobs and endpoint concepts

---

## Milestones

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Configure Spot Training](console_guides/01_configure_spot_training.md) | 25 min | A spot-enabled training job with checkpointing |
| 2 | [Right-Size Instances](console_guides/02_right_size_instances.md) | 20 min | A right-sizing analysis with CloudWatch metrics |
| 3 | [Configure Auto-Scaling](console_guides/03_configure_autoscaling.md) | 25 min | A target-tracking scaling policy on an endpoint |
| 4 | [Run Inference Recommender](console_guides/04_run_inference_recommender.md) | 25 min | An Inference Recommender comparison report |
| 5 | [Build CloudWatch Dashboard](console_guides/05_build_cloudwatch_dashboard.md) | 25 min | A multi-widget operational dashboard |
| 6 | [Cleanup](console_guides/06_cleanup.md) | 15 min | All resources deleted |

**Total estimated time:** ~135 minutes

---

## Presentation Deliverables
1. Screenshot of the completed spot training job showing ManagedSpotTrainingEnabled and savings percentage
2. Screenshot of CloudWatch CPU/memory graphs for the training job with your right-sizing recommendation documented
3. Screenshot of the auto-scaling policy configuration showing target metric, min, and max instance counts
4. Screenshot of the Inference Recommender results table comparing instance types by latency, throughput, and cost
5. Screenshot of the completed CloudWatch dashboard with all four or more widgets visible
6. Screenshot confirming all dashboards, scaling policies, endpoints, and S3 artifacts have been deleted

---

## Important Reminders
- **Free Tier:** Use ml.m5.xlarge or smaller. No GPU instances.
- **Region Consistency:** Stay in us-east-1.
- **Cleanup Is Mandatory:** Always complete the cleanup guide.
- **Do Not Skip Steps:** Each guide builds on the previous one.
