# AWSSageMaker-MLOps Lab

## Scenario

FraudShield's fraud detection model is now being retrained regularly as new transaction data arrives. Manual training, registration, and deployment cycles are becoming unsustainable. Your job is to automate the ML workflow using SageMaker Pipelines -- building a DAG that preprocesses data, trains a model, evaluates it against a quality threshold, and conditionally registers it in the Model Registry.

This lab continues from Modules 2 and 3 -- you will use the same Studio Domain, S3 bucket, and model registry group.

---

## Learning Objectives

By completing this lab you will demonstrate the ability to:

1. Explain the core principles of MLOps and how CI/CD adapts for machine learning
2. Build a SageMaker Pipeline with multiple steps (Processing, Training, Registration)
3. Monitor a pipeline execution in the console using the DAG visualization
4. Add a quality gate using a ConditionStep that evaluates model metrics
5. Understand where Model Monitor and EventBridge fit in the MLOps landscape

---

## Prerequisites

- Completed Modules 1-3 labs (Studio Domain, S3 bucket, Model Registry group in place)
- Familiarity with SageMaker Pipelines concepts from the readings
- Understanding of DAGs (Directed Acyclic Graphs) from the readings

---

## Milestones

| # | Guide | Estimated Time | What You Do |
|---|-------|---------------|-------------|
| 1 | [Understand MLOps in the Console](console_guides/01_understand_mlops_console.md) | 15 min | Tour the MLOps-related console sections |
| 2 | [Build and Run a SageMaker Pipeline](console_guides/02_build_and_run_pipeline.md) | 30 min | Create and execute a multi-step pipeline |
| 3 | [Monitor Pipeline Execution in the Console](console_guides/03_monitor_pipeline_execution.md) | 20 min | Watch the DAG, explore step details |
| 4 | [Add a Quality Gate with ConditionStep](console_guides/04_add_quality_gate.md) | 25 min | Add evaluation + conditional branching |
| 5 | [Explore Model Monitor and EventBridge](console_guides/05_explore_monitor_eventbridge.md) | 15 min | Conceptual tour of monitoring and automation |

**Total estimated time:** ~105 minutes

---

## Presentation Deliverables

When presenting your work, be prepared to show and explain:

1. The MLOps-related sections of the SageMaker console and what each one does
2. Your pipeline definition and the DAG visualization in the console
3. A completed pipeline execution with all steps showing success (green)
4. The ConditionStep logic and how it creates branching in the DAG
5. Where Model Monitor and EventBridge fit in the MLOps lifecycle
6. A model version that was automatically registered by the pipeline

---

## Important Reminders

- **No Persistent Endpoints:** This lab does NOT deploy endpoints. Pipeline steps (Processing, Training) run on ephemeral instances that terminate automatically after each step completes. You will not incur ongoing charges.
- **S3 Costs:** Pipeline outputs (processed data, model artifacts, evaluation reports) are stored in S3 with minimal storage costs.
- **Region Consistency:** Use the same region as all previous modules.
