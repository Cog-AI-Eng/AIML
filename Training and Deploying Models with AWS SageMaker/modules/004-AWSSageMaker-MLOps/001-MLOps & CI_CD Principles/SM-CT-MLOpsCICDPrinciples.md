# MLOps & CI/CD Principles

**Estimated Time:** 10 Minutes

## Introduction

Over the last three modules you completed the SageMaker ML lifecycle manually: you set up Studio, configured IAM roles, trained models with Script Mode, stored artifacts in S3, registered versions in the Model Registry, deployed endpoints, and invoked them for predictions. Each step worked. But you did every step by hand -- clicking through consoles, running SDK calls in notebooks, copying S3 paths, and remembering to clean up resources.

That manual approach works for learning and for one-off experiments. It does not work when your team needs to retrain a model every week on fresh data, deploy the new version if it passes quality checks, roll back if monitoring detects degradation, and do all of this without a human babysitting every step. That is the problem **MLOps** solves.

If you recall the *ML Lifecycle & Reproducibility* reading from the AIML Foundations module, you learned three pillars of reproducibility: consistent environments, version-controlled code, and fixed random seeds. MLOps takes those pillars and adds a fourth: **automation**. MLOps is the practice of applying software engineering's CI/CD (Continuous Integration / Continuous Delivery) principles to machine learning workflows so that the entire lifecycle -- from data ingestion to deployment to monitoring -- runs reliably and repeatably without manual intervention.

This reading introduces the core MLOps principles, maps them to the SageMaker tools you already know, and shows where these tools live in the console.

## Core Concepts

### What MLOps means

MLOps is not a single tool or service. It is a discipline that combines three practices:

**Automation** replaces manual steps with orchestrated workflows. Instead of you clicking "Create training job" in the console, a pipeline triggers training automatically when new data arrives or on a scheduled cadence.

**Versioning and governance** ensures that every artifact (data, code, model, configuration) is tracked and every transition (from training to registry to deployment) is auditable. You have already practiced this with Git for code and the Model Registry for models. MLOps formalizes it across the entire pipeline.

**Monitoring and feedback** closes the loop. Once a model is deployed, monitoring systems watch for data drift, model quality degradation, and infrastructure issues. When monitoring detects a problem, it triggers the pipeline to retrain, re-evaluate, and re-deploy -- completing the cycle without human intervention for routine cases.

### CI/CD adapted for machine learning

In traditional software engineering, CI/CD has a clear meaning:

- **Continuous Integration (CI):** Every code change is automatically built, tested, and merged.
- **Continuous Delivery (CD):** Every successful build is automatically deployable to production.

Machine learning adds complexity because the "product" is not just code -- it is code *plus data plus a trained model*. The CI/CD concepts adapt as follows:

| Software CI/CD | ML CI/CD (MLOps) | SageMaker Tool |
| :--- | :--- | :--- |
| Code commit triggers build | Code or data change triggers training pipeline | SageMaker Pipelines, EventBridge |
| Unit tests validate code | Model evaluation validates quality | Evaluation step in pipeline, Model Registry metrics |
| Build artifact (binary) | Model artifact (model.tar.gz) | S3, Model Registry |
| Deploy to staging/production | Deploy to endpoint (staging, then production) | Endpoint deployment from Registry |
| Application monitoring | Model monitoring (data drift, quality drift) | SageMaker Model Monitor |
| Bug fix triggers new build | Drift alert triggers retraining | EventBridge + Pipeline trigger |

The key insight is that ML pipelines have **two sources of change**: code changes (you update the training script or features) and data changes (the input distribution shifts over time). Both can degrade model quality, and both should trigger the retraining pipeline.

### The MLOps maturity spectrum

Teams typically progress through levels of automation:

**Level 0 -- Manual.** Every step is executed by hand. This is how you have worked through Modules 1-3. It is appropriate for learning and prototyping but does not scale.

**Level 1 -- Pipeline automation.** Training, evaluation, and registration are automated in a pipeline. A human still reviews and approves model versions before deployment. This is the level most teams target first.

**Level 2 -- Full CI/CD.** Code changes trigger pipeline execution automatically. Model evaluation gates deployment without human intervention for routine cases. Monitoring triggers retraining. Humans intervene only for exceptions.

For this curriculum, you will build Level 1 workflows: automated pipelines with manual approval gates. Level 2 is a stretch goal.

### SageMaker tools for MLOps

You already know most of the tools. Here is how they fit into the MLOps picture:

**SageMaker Pipelines** is the orchestration engine. A pipeline defines a sequence of steps (preprocessing, training, evaluation, registration, deployment) as a directed acyclic graph (DAG). When the pipeline runs, SageMaker executes each step in order, passing outputs from one step as inputs to the next. In the console sidebar, pipelines live under **Pipelines > Pipelines**. The next reading covers Pipelines in detail.

**SageMaker Model Registry** (Module 3) is the versioning and governance layer. Pipelines register new model versions automatically; approval workflows gate deployment. You have already used the Registry manually -- in an MLOps workflow, the pipeline interacts with it programmatically.

**SageMaker Model Monitor** detects drift in production. After you deploy an endpoint, Model Monitor compares incoming request data against a baseline (the data the model was trained on) and flags statistical shifts. It also tracks model quality by comparing predictions against ground truth when available. In the console, Model Monitor lives under **Inference > Model monitoring**.

Model Monitor addresses a question you may have had since the AIML Evaluation module: "how do I know if my model's accuracy is degrading after deployment?" The answer is continuous monitoring. If your training data represented customer behavior in January, but by March customer behavior has shifted, the model's predictions may become less accurate even though the code has not changed. Model Monitor catches this kind of **data drift** and **concept drift** and can trigger retraining through EventBridge.

**Amazon EventBridge** (introduced in the *Approval Workflows* reading) connects events to actions. A Model Monitor alert can trigger a Pipeline execution. A new data upload to S3 can trigger preprocessing. A model approval status change can trigger deployment. EventBridge is the glue that makes the feedback loop automatic.

**AWS CodePipeline and CodeBuild** handle the CI side: when you push code changes to a Git repository (CodeCommit, GitHub, or Bitbucket), CodePipeline triggers CodeBuild to run tests and then kicks off the SageMaker Pipeline. This connects your version-controlled training code to the automated ML pipeline. In the console, search "CodePipeline" to find the CI/CD service.

### Where MLOps tools live in the console

| Tool | Console Navigation |
| :--- | :--- |
| SageMaker Pipelines | **SageMaker > Pipelines > Pipelines** |
| Model Registry | **SageMaker > Governance > Model registry** |
| Model Monitor | **SageMaker > Inference > Model monitoring** |
| EventBridge | Search "EventBridge" > **Rules** |
| CodePipeline | Search "CodePipeline" > **Pipelines** |
| CodeBuild | Search "CodeBuild" > **Build projects** |

You do not need to visit all of these today. The table is a reference map, similar to the lifecycle-to-console table from Module 1. When you reach each tool in practice, you will know where to find it.

### Model Monitor: detecting drift (conceptual overview)

Model Monitor works in four steps:

1. **Baseline.** You provide a baseline dataset (typically the training data) and Model Monitor computes statistical properties: feature distributions, means, standard deviations, and data types.
2. **Data capture.** You enable data capture on your endpoint (an option in the Endpoint Configuration, as mentioned in the *Real-time Inference Endpoints* reading). SageMaker logs a configurable percentage of incoming requests and responses to S3.
3. **Monitoring schedule.** You create a monitoring job that runs on a schedule (hourly, daily). The job compares captured data against the baseline and generates a violations report if statistical properties have drifted beyond thresholds.
4. **Alerts.** If violations are detected, Model Monitor emits events to EventBridge. You can configure these events to send notifications (SNS), trigger a retraining pipeline, or flag the model for human review.

The detailed setup of Model Monitor is covered in the module assignment and lecture. This reading introduces it so you understand its role in the MLOps feedback loop.

### SDK perspective

The SageMaker Python SDK and `boto3` provide programmatic access to all MLOps tools. Pipelines are defined in Python code using the `sagemaker.workflow` module. Model Monitor is configured through the `sagemaker.model_monitor` module. EventBridge rules are created through `boto3`'s `events` client. You will write pipeline code in the next reading and in the module assignment.

## Connecting to Practice

This reading gives you the conceptual framework for MLOps. In the *MLOps & CI/CD Principles Video*, you will see an overview of an automated pipeline running end to end. The next reading, *Pipelines, DAGs & Versioning*, teaches you how to build a SageMaker Pipeline. And in the module lecture and assignment, you will construct a pipeline that automates the train-evaluate-register cycle you have been doing manually.

The most useful thing you can do right now is open the SageMaker console and navigate to **Pipelines > Pipelines**. If the list is empty, that is expected -- you have not created one yet. Also visit **Inference > Model monitoring** to see the monitoring interface. Knowing where these tools live sets you up for the hands-on work ahead.

---

## Further Learning & Resources

**Documentation and reading**

- **[MLOps with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-whatis.html)** - *Docs*: The official overview of SageMaker's MLOps capabilities, covering projects, pipelines, model registry, and monitoring.
- **[Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)** - *Docs*: The complete guide to setting up baselines, data capture, monitoring schedules, and violation detection.

**Interactive practice**

- **[SageMaker MLOps Workshop](https://catalog.workshops.aws/sagemaker-mlops/en-US)** - *Interactive*: A comprehensive workshop covering the full MLOps lifecycle with SageMaker Pipelines, Model Registry, and Model Monitor.
- **[AWS Skill Builder - MLOps Engineering on AWS](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/18970/mlops-engineering-on-aws)** - *Interactive*: Self-paced training covering CI/CD for ML, pipeline automation, and monitoring patterns.
