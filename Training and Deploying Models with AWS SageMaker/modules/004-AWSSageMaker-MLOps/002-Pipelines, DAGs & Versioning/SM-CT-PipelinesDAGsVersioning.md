# Pipelines, DAGs & Versioning

**Estimated Time:** 10 Minutes

## Introduction

In the *MLOps & CI/CD Principles* reading you learned that MLOps replaces manual steps with automated workflows. The tool SageMaker provides for this automation is **SageMaker Pipelines** -- a service that lets you define, execute, and manage multi-step ML workflows as code.

Think back to the ML lifecycle you studied in AIML Foundations: data ingestion, preprocessing, training, evaluation, and deployment. You executed each of those stages manually throughout this curriculum -- running processing jobs, training jobs, registering models, deploying endpoints. A SageMaker Pipeline encodes that exact sequence into a reusable definition so SageMaker can run the whole thing automatically, end to end, every time you need a new model.

The pipeline definition takes the form of a **Directed Acyclic Graph (DAG)** -- a set of steps connected by dependencies, where each step's output feeds the next step's input, and the flow moves in one direction without looping back. If that sounds abstract, picture a recipe card: step 1 (chop vegetables) must finish before step 2 (saute them), which must finish before step 3 (add to broth). The DAG is the recipe card for your ML workflow.

This reading explains how to define pipeline steps, view them in the console, execute pipelines, and introduces advanced features like parameterization and conditional logic.

## Core Concepts

### Pipeline steps

A SageMaker Pipeline is a sequence of **steps**, where each step performs one action. The most common step types map directly to the SageMaker operations you already know:

| Step Type | What It Does | You Already Know This As... |
| :--- | :--- | :--- |
| `ProcessingStep` | Runs a data preprocessing or evaluation script | SageMaker Processing Jobs |
| `TrainingStep` | Runs a training job using an Estimator | `estimator.fit()` |
| `CreateModelStep` | Creates a Model object from a training artifact | Creating a Model in the console |
| `RegisterModel` | Registers a model version in the Model Registry | Registering a version in the Registry |
| `ConditionStep` | Branches the pipeline based on a condition | New (decision logic) |
| `TransformStep` | Runs a batch transform job | Batch Transform |

Each step wraps the configuration you would normally pass to the SageMaker API or SDK. The difference is that instead of executing immediately, the step definition is saved as part of the pipeline and executed when the pipeline runs.

### Building a simple pipeline in code

Here is a minimal pipeline that preprocesses data, trains a model, and registers it. You define this in a Python script or notebook:

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString

input_data = ParameterString(name="InputData", default_value="s3://my-bucket/data/raw/")

preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination="s3://my-bucket/data/processed/")],
    code="preprocess.py",
)

train_step = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs={"train": preprocess_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri},
)

register_step = RegisterModel(
    name="Register",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    approval_status="PendingManualApproval",
    model_package_group_name="fraud-detection-rf",
)

pipeline = Pipeline(
    name="fraud-detection-pipeline",
    parameters=[input_data],
    steps=[preprocess_step, train_step, register_step],
)
```

Notice how each step references the previous step's output. `train_step` reads data from `preprocess_step`'s output S3 path. `register_step` reads the model artifact from `train_step`'s output. These references create the DAG automatically -- SageMaker knows the execution order based on the data dependencies.

This code does not run the pipeline. It *defines* the pipeline. To make it available for execution, you submit it:

```python
pipeline.upsert(role_arn=role)
```

This creates (or updates) the pipeline definition in SageMaker. It now appears in the console and can be executed on demand or on a schedule.

### Pipeline parameters

The `ParameterString` in the example above (`InputData`) makes the pipeline reusable. Instead of hardcoding the S3 path for raw data, you define it as a parameter with a default value. When you execute the pipeline, you can override the default:

```python
pipeline.start(parameters={"InputData": "s3://my-bucket/data/march-2026/"})
```

This means the same pipeline definition can process different datasets without editing code. Parameters can be strings, integers, or floats. Common parameters include input data paths, hyperparameter values, instance types, and approval status defaults.

### Viewing pipelines in the console

Once a pipeline is created, you can manage it entirely from the browser.

1. **Navigate to Pipelines.** In the SageMaker sidebar, click **Pipelines > Pipelines**. You see a list of all defined pipelines with their names, creation dates, and last execution status.

2. **Click a pipeline name** to open its details page. You will see:
   - **Graph tab:** A visual DAG showing each step as a node and the dependencies as arrows. This is the most intuitive view of your pipeline. Preprocess flows into Train, which flows into Register. The layout updates automatically based on the step definitions in your code.
   - **Parameters tab:** Lists all parameters with their default values. You can see what inputs the pipeline expects.
   - **Executions tab:** Shows the history of all pipeline runs with status (Executing, Succeeded, Failed), start time, and duration.

3. **Click an execution** to drill into its details:
   - Each step shows its own status (Succeeded, Failed, Executing).
   - Click a step to see its configuration, input/output paths, logs (linked to CloudWatch), and duration.
   - If a step failed, the error message and log link help you debug without leaving the console.

> **Tip:** The visual DAG in the console is the fastest way to understand a pipeline's structure, especially for pipelines you did not write yourself. Spend time reading DAGs -- they are the blueprints of production ML systems.

### Executing a pipeline from the console

1. On the pipeline details page, click **Create execution** (or **Start execution**).
2. **Review parameters.** The form shows all parameters with their default values. Override any you need to change for this run.
3. **Click Start.** The execution begins and you can track progress in the Executions tab.
4. **Monitor each step.** Click into the running execution to see the DAG with real-time status updates. Green nodes have completed; blue nodes are running; red nodes have failed.

### Conditional logic (advanced)

For more sophisticated workflows, `ConditionStep` lets the pipeline branch based on runtime values. A common pattern is evaluating model quality and only registering the model if it meets a threshold:

```python
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name="Evaluate", property_file=eval_report, json_path="metrics.f1"),
    right=0.85,
)

condition_step = ConditionStep(
    name="CheckQuality",
    conditions=[condition],
    if_steps=[register_step],
    else_steps=[],
)
```

This step reads the F1 score from an evaluation report (generated by an earlier evaluation step), compares it to 0.85, and only proceeds to registration if the condition passes. If the condition fails, the pipeline ends without registering a low-quality model.

This pattern connects directly to the approval criteria framework from the *Approval Workflows* reading. The condition step automates the quality check that you previously performed manually.

### Pipeline versioning

Every time you call `pipeline.upsert()`, SageMaker updates the pipeline definition. However, past executions retain the definition they ran with, so you can always inspect what a previous run actually executed. This is implicit versioning: the pipeline definition can evolve while historical runs remain frozen in their original state.

For explicit versioning, teams often store pipeline definition code in Git alongside their training scripts. Each Git commit captures a specific pipeline configuration, and CI/CD tools (like CodePipeline) can trigger `pipeline.upsert()` on every merge to main. This connects the pipeline versioning story to the code versioning practices from the AIML Foundations module.

### Connecting to CI/CD with CodePipeline (conceptual)

For teams practicing Level 2 MLOps, AWS CodePipeline can automate the entire flow:

1. A developer pushes a code change to a Git repository.
2. CodePipeline detects the change and triggers CodeBuild.
3. CodeBuild runs tests on the training script and calls `pipeline.upsert()` to update the SageMaker Pipeline definition.
4. CodeBuild then calls `pipeline.start()` to execute the pipeline.
5. The pipeline trains, evaluates, and registers the model.
6. A human (or automated gate) approves the model in the Registry.
7. A deployment step (or a separate CodePipeline stage) deploys the approved model to an endpoint.

This end-to-end automation is a stretch goal for this curriculum. The important concept is that SageMaker Pipelines and CodePipeline serve different roles: SageMaker Pipelines orchestrates ML steps (training, evaluation, registration), while CodePipeline orchestrates software delivery steps (build, test, deploy). They work together in a mature MLOps setup.

## Connecting to Practice

This reading completes the curriculum's content. You now have the conceptual and practical knowledge to build automated ML workflows on SageMaker. In the *Pipelines, DAGs & Versioning Video*, you will see a pipeline executed and monitored in the console. In the module lecture, you will build and run a pipeline. And in the module assignment, you will create a multi-step pipeline that automates the training-to-registration workflow you have been doing by hand.

The most useful thing you can do right now is open the SageMaker console and navigate to **Pipelines > Pipelines**. If you have not created a pipeline yet, review the code example in this reading and think about how it maps to the manual steps you performed in Modules 2 and 3. Each step in the pipeline replaces a manual click or SDK call you already know.

---

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)** - *Docs*: The official guide covering pipeline definition, step types, parameters, conditions, and execution management.
- **[SageMaker Pipeline Step Types](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html)** - *Docs*: Detailed reference for every available step type with parameter specifications and examples.

**Interactive practice**

- **[SageMaker Pipelines Workshop](https://catalog.workshops.aws/sagemaker-pipelines/en-US)** - *Interactive*: A hands-on workshop walking through pipeline creation, execution, and monitoring in your own AWS account.
- **[SageMaker Examples - Pipelines](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-pipelines)** - *Interactive*: Runnable notebooks demonstrating multi-step pipelines with processing, training, evaluation, and conditional registration.
