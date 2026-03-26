# Experiment Tracking and Lineage Lab

## Scenario
FraudShield Risk Analytics has trained multiple XGBoost models with different hyperparameter configurations. The data-science team now faces a governance challenge: which model was trained on which version of the data, with what parameters, and where is the resulting artifact stored? Without answers to these questions, the team cannot reproduce results, satisfy audit requirements, or confidently promote a model to production.

In this lab you will use SageMaker Experiments to organize and compare training runs, then use the Lineage Graph to trace a model artifact back through every step that produced it -- from the original dataset through the training job to the final model. You will finish by compiling a reproducibility report that documents everything needed to recreate a specific model from scratch.

---

## Learning Objectives
By completing this lab you will demonstrate the ability to:
1. Create a SageMaker Experiment and associate multiple training runs with different hyperparameters.
2. Use the Experiments comparison chart to filter, sort, and identify the best-performing run.
3. Navigate the Lineage Graph for a trained model and identify all entity types (Context, Action, Artifact, Association).
4. Build a reproducibility report that documents the exact data version, hyperparameters, environment, and model artifact location.
5. Clean up all experiment and training job resources.

---

## Prerequisites
- A SageMaker Studio Domain provisioned in us-east-1 with an active user profile.
- The FraudShield training and validation datasets uploaded to S3 in CSV format.
- Familiarity with SageMaker training jobs (completed Module 2 or equivalent).
- An IAM execution role with S3, SageMaker, and CloudWatch permissions.

---

## Milestones

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Create an Experiment](console_guides/01_create_experiment.md) | 30 min | An experiment with 3 training runs using different hyperparameters |
| 2 | [Compare Experiment Runs](console_guides/02_compare_experiment_runs.md) | 20 min | A side-by-side comparison chart identifying the best run |
| 3 | [Explore the Lineage Graph](console_guides/03_explore_lineage_graph.md) | 20 min | A traced lineage path from endpoint to dataset |
| 4 | [Build a Reproducibility Report](console_guides/04_build_reproducibility_report.md) | 20 min | A documented report with data version, parameters, and artifact location |
| 5 | [Cleanup](console_guides/05_cleanup.md) | 10 min | All experiment runs, experiments, and training artifacts deleted |
| SDK | [SDK Experiments Lab](notebooks/sdk_experiments_lab.ipynb) | 45 min | Create Experiments, log metrics with the Run API, query lineage programmatically, and build a reproducibility report using the SageMaker Python SDK |

**Total estimated time:** ~145 minutes (console guides ~100 min + SDK notebook ~45 min)

---

## Presentation Deliverables
1. Show the Experiment in the console with all three training runs listed.
2. Present the comparison chart with metric filters applied and the best run highlighted.
3. Walk through the Lineage Graph for the best model, naming each entity type encountered.
4. Present the reproducibility report with all required fields populated.
5. Confirm all resources have been deleted in the cleanup guide.

---

## Important Reminders
- **Free Tier:** Use ml.m5.xlarge or smaller. No GPU instances.
- **Region Consistency:** Stay in us-east-1.
- **Cleanup Is Mandatory:** Always complete the cleanup guide.
- **Do Not Skip Steps:** Each guide builds on the previous one.
