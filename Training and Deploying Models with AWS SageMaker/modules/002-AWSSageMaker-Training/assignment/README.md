# AWSSageMaker-Training Lab

## Scenario

The FraudShield data science team has built a fraud detection model locally using scikit-learn. Your job is to migrate this training workflow to SageMaker's managed infrastructure. In this lab, you will prepare training data in S3, explore how the console maps to SDK parameters, launch a real training job, monitor it, inspect the resulting artifacts, and debug a failed job.

This lab continues from Module 1 -- you will use the Studio Domain and IAM roles you already created.

---

## Learning Objectives

By completing this lab you will demonstrate the ability to:

1. Create an S3 bucket and organize training data with proper prefix structure
2. Map every field in the SageMaker "Create Training Job" console form to its SDK equivalent
3. Launch a training job from a Studio notebook using the SageMaker SDK
4. Monitor a training job's lifecycle stages in the console and CloudWatch Logs
5. Locate and understand model artifacts stored in S3 after training
6. Debug a failed training job using console details and CloudWatch error logs

---

## Prerequisites

- Completed Module 1 lab (Studio Domain and IAM roles in place)
- An S3 bucket for training data (you will create one in Guide 1)
- Familiarity with Script Mode structure from the readings

---

## Milestones

| # | Guide | Estimated Time | What You Do |
|---|-------|---------------|-------------|
| 1 | [Prepare Training Data in S3](console_guides/01_prepare_training_data.md) | 15 min | Create bucket, upload CSV data |
| 2 | [Explore the Create Training Job Form](console_guides/02_explore_training_job_form.md) | 20 min | Map every console field to SDK parameters |
| 3 | [Launch a Training Job from Studio](console_guides/03_launch_training_job.md) | 20 min | Run a real training job via notebook |
| 4 | [Monitor and Inspect the Training Job](console_guides/04_monitor_training_job.md) | 20 min | Observe lifecycle, check CloudWatch Logs |
| 5 | [Find and Verify Model Artifacts](console_guides/05_find_model_artifacts.md) | 15 min | Navigate S3 to find model.tar.gz |
| 6 | [Debug a Failed Training Job](console_guides/06_debug_failed_job.md) | 15 min | Launch a job with an intentional error, diagnose it |

**Total estimated time:** ~105 minutes

---

## Presentation Deliverables

When presenting your work, be prepared to show and explain:

1. Your S3 bucket with organized training data (prefix structure)
2. The console form field-to-SDK mapping you documented
3. Your completed training job in the console (status, duration, instance type)
4. CloudWatch Logs showing training output
5. The model artifact location in S3 and the path pattern
6. The failed job, its error message, and your diagnosis of the root cause

---

## Important Reminders

- **Free Tier:** Use `ml.m5.xlarge` for training instances. Do not select GPU instances.
- **Region Consistency:** Use the same region as Module 1 (`us-east-1`).
- **Training Jobs Are Ephemeral:** Unlike endpoints, training job instances are automatically terminated when the job completes. You will NOT be charged after the job finishes.
- **S3 Storage:** The training data and model artifacts stored in S3 incur minimal storage costs and should be kept for Module 3.
