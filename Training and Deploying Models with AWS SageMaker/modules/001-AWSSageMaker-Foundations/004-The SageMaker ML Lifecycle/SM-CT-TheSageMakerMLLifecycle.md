# The SageMaker ML Lifecycle

**Estimated Time:** 10 Minutes

## Introduction

In the *Ecosystem & Core Services* reading, you saw a table mapping general ML lifecycle stages to SageMaker services and their locations in the console. That table gave you the bird's-eye view. This reading zooms in and walks through the five stages that AWS uses to organize the SageMaker experience: **Prepare, Build, Train & Tune, Deploy, and Monitor**. These five stages are not a different lifecycle from what you learned in the Applied ML Foundations module -- they are the same fundamental loop (data in, model out, watch it run) repackaged around the tools SageMaker provides.

Why does the repackaging matter? Because when you open the SageMaker console, the sidebar, the documentation, and the SDK are all organized around these five stages. If you understand the framework, you can predict where to find any tool and how it fits into your workflow. You stop guessing which menu to click and start navigating with intent.

By this point you have already set up your workspace (Studio Domains & Profiles) and secured your permissions (IAM & Least-Privilege Practices). Those are prerequisites for the lifecycle. Now you are ready to understand the full journey that your data and models will take through SageMaker.

## Core Concepts

### The five stages at a glance

SageMaker groups its services into five lifecycle stages. Here is how they map to the general ML lifecycle you studied in the AIML Foundations content:

| SageMaker Stage | What Happens | AIML Lifecycle Equivalent | Console Location |
| :--- | :--- | :--- | :--- |
| **Prepare** | Collect, clean, label, and transform data | Data ingestion, EDA, feature engineering | **SageMaker > Processing**, **SageMaker > Ground Truth**, **S3 console** |
| **Build** | Write and iterate on code in notebooks; select algorithms | Problem framing, model selection, experimentation | **SageMaker > Studio**, **SageMaker > Notebook instances** |
| **Train & Tune** | Run training jobs on managed compute; optimize hyperparameters | Model training, evaluation | **SageMaker > Training > Training jobs**, **SageMaker > Training > Hyperparameter tuning jobs** |
| **Deploy** | Package models and serve predictions via endpoints | Deployment | **SageMaker > Inference > Models**, **SageMaker > Inference > Endpoints** |
| **Monitor** | Track data quality, model quality, and endpoint health | Monitoring, iteration | **SageMaker > Inference > Model monitoring**, **SageMaker > Experiments** |

The AIML lifecycle taught you *why* each stage exists. This table shows you *where* SageMaker puts the tools for each one. Open the SageMaker console now and match each sidebar section to a row in this table -- you will recognize the structure immediately.

### Stage 1: Prepare

Before any model can learn, your data needs to be in the right place and the right shape. In the general lifecycle, this covers data ingestion, exploratory data analysis, and feature engineering. SageMaker maps these to three tools:

**Amazon S3** is the storage layer. Raw datasets, processed features, and eventually model artifacts all live in S3 buckets. In the *Ecosystem* reading you learned how to find S3 in the console and name your buckets. During the Prepare stage, you upload your raw data here.

**SageMaker Processing** lets you run data transformation scripts (cleaning, splitting, feature computation) on managed compute instances. Instead of running a preprocessing script on your laptop and hoping it finishes, you describe the job and SageMaker provisions the compute, runs it, and writes the output back to S3. In the console sidebar, look under **Processing > Processing jobs** to see past and active jobs.

**SageMaker Ground Truth** is a data labeling service. If your dataset needs human-annotated labels (image classifications, entity tags, sentiment labels), Ground Truth provides labeling workflows with built-in quality controls. For structured tabular data that is already labeled, you may skip Ground Truth entirely. You can find it under **Ground Truth** in the sidebar.

The Prepare stage ends when you have a clean, labeled dataset stored in S3 and ready for training.

### Stage 2: Build

Building is where experimentation happens. You open SageMaker Studio (which you configured in the *Studio Domains & Profiles* topic), create notebooks, load your prepared data, and write the code that will become your training script.

In the AIML Foundations content, this corresponds to problem framing and model selection -- deciding which algorithm to try, setting up your train/test splits, and writing the training logic. The key difference in SageMaker is that your notebook runs in a managed cloud environment (Studio) rather than on your local machine, so you get consistent compute, shared storage on EFS, and seamless access to other AWS services through the execution role you configured in the *IAM* reading.

You can also explore **SageMaker JumpStart** during the Build stage to browse pre-built models and solution templates before writing anything from scratch. JumpStart has its own dedicated topic next in this module, so we will cover it there.

In the console, the Build stage lives primarily in **SageMaker > Studio**. If you are using classic notebook instances instead of Studio, they appear under **Notebook > Notebook instances**.

### Stage 3: Train & Tune

Once your training script is ready, you move from experimentation to execution. SageMaker Training takes your script, your data location in S3, and your configuration (instance type, hyperparameters, output path) and runs the job on dedicated, managed infrastructure. The compute spins up, trains your model, writes the resulting model artifact (a compressed `model.tar.gz` file) to S3, and shuts down automatically.

This maps to the *model training* and *evaluation* stages from the AIML lifecycle. The difference is scale and governance: SageMaker handles the infrastructure, logs metrics to CloudWatch, and records the job details so you can compare runs later.

**Hyperparameter tuning jobs** take training a step further. Instead of manually trying different learning rates or batch sizes, you define ranges and SageMaker runs multiple training jobs in parallel, searching for the combination that optimizes your chosen metric. In the console, tuning jobs appear under **Training > Hyperparameter tuning jobs**.

In the console, training jobs are under **Training > Training jobs**. Click any completed job to see its configuration, input/output paths, metrics, and logs. The *Training Job Anatomy* topic in Module 2 will walk through these details, so this reading stays at the overview level.

### Stage 4: Deploy

A trained model sitting in S3 is useful only if something can serve predictions from it. The Deploy stage is where you take a model artifact, package it into a deployable format, and host it behind an endpoint.

SageMaker offers several deployment options:

- **Real-time endpoints** serve predictions synchronously with low latency. You create a model object, define an endpoint configuration (instance type, instance count), and launch an endpoint. Clients send requests and get responses in milliseconds.
- **Batch transform** processes large datasets offline. Instead of one prediction at a time, you point SageMaker at an S3 input file and it processes every record, writing results to an output location.
- **Serverless inference** provisions compute on demand and scales to zero when idle, which is useful for workloads with infrequent traffic.

In the console, deployments live under **Inference**. You will see submenus for **Models**, **Endpoint configurations**, and **Endpoints**. The *Real-time Inference Endpoints* and *Invoking Endpoints* topics in Module 3 cover deployment in depth. The *Model Registry & Versioning* topic in that same module covers how to track which model version is behind which endpoint.

### Stage 5: Monitor

Deploying a model is not the end of the lifecycle -- it is the beginning of the feedback loop. Once a model serves live traffic, you need to watch for **data drift** (the input data distribution shifts away from what the model was trained on) and **model quality degradation** (accuracy or other metrics decline over time).

SageMaker Model Monitor automates this by comparing incoming request data and prediction outputs against a baseline you define. When drift exceeds a threshold, Model Monitor raises an alert. In the console, monitoring lives under **Inference > Model monitoring**.

This stage connects back to the AIML lifecycle's iteration principle: monitoring tells you when it is time to retrain, refine features, or revisit your problem framing. The cycle begins again.

### How the stages connect

The five stages are not a straight line. They form a loop:

1. **Prepare** data and store it in S3.
2. **Build** your training logic in Studio notebooks.
3. **Train & Tune** on managed compute, producing model artifacts in S3.
4. **Deploy** the best model to an endpoint.
5. **Monitor** the endpoint for drift or degradation.
6. When monitoring signals a problem, return to **Prepare** or **Build** and iterate.

Every service you have seen in earlier readings -- S3, Studio, IAM roles, the console sidebar -- fits into one or more of these stages. The five-stage framework is simply the organizing principle that tells you where you are in the loop and which SageMaker tools to reach for next.

### SDK and CLI equivalents

Each stage has corresponding SDK and CLI commands: `sagemaker.processing.Processor` for Prepare, `sagemaker.estimator.Estimator` for Train, `model.deploy()` for Deploy, and `sagemaker.model_monitor` for Monitor. You will use these extensively in later modules. As with previous topics, the console workflow comes first so you understand the underlying concepts before automating them.

## Connecting to Practice

This reading gives you the complete mental model for how SageMaker organizes the ML lifecycle. In the upcoming video, you will see these stages demonstrated in the console. In the next topic, *JumpStart Pre-built Models*, you will see how SageMaker accelerates the Build stage by offering ready-made models. And in the module lecture and assignment, you will walk through the full five-stage loop from Prepare to Monitor.

The most useful thing you can do right now is open the SageMaker console and click through each sidebar section. For each section, identify which lifecycle stage it belongs to. If you can do that without referring back to the table above, you have internalized the framework and are ready for the hands-on modules ahead.

---

## Further Learning & Resources

**Documentation and reading**

- **[How Amazon SageMaker Works](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works.html)** - *Docs*: The official architecture overview showing how SageMaker's services connect across the ML lifecycle, useful as a reference when you encounter new features.
- **[Machine Learning Workflow on AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-mlconcepts.html)** - *Docs*: AWS's mapping of ML concepts to SageMaker tools, reinforcing the lifecycle framework with additional detail on each stage.

**Interactive practice**

- **[AWS Hands-On: Build, Train, and Deploy a Machine Learning Model](https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)** - *Interactive*: A free end-to-end lab that walks you through all five lifecycle stages in your own AWS console.
- **[AWS Skill Builder - SageMaker Immersion Day](https://catalog.workshops.aws/sagemaker-immersion-day/en-US)** - *Interactive*: A workshop-style environment with guided exercises covering data preparation, training, and deployment on SageMaker.
