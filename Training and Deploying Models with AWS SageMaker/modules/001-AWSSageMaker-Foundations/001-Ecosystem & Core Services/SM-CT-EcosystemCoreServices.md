# Ecosystem & Core Services

**Estimated Time:** 10 Minutes

## Introduction

If you have already started the Applied ML Foundations content, you know the standard machine learning lifecycle: frame the problem, ingest data, explore and clean, engineer features, train, evaluate, deploy, and monitor. Those stages are universal. What changes from team to team is the *infrastructure* underneath them. Running a training script on your laptop works for toy datasets, but the moment you need larger data, longer training runs, or a model that serves real traffic, you need an environment that can scale without requiring you to wire up servers by hand.

AWS SageMaker is Amazon's fully managed platform for exactly that problem. Think of it as a purpose-built workshop that gives you a room for every stage of the ML lifecycle: a workbench for exploration, storage racks for data, a high-powered forge for training, and a service window for deploying models to production. The individual rooms are separate AWS services, but SageMaker ties them together so you can move from notebook to endpoint without leaving the ecosystem.

This reading introduces the major services in that ecosystem and shows you where to find them in the AWS Management Console. The goal is for you to feel comfortable opening the console, navigating to SageMaker, and identifying the tools you will use throughout this skill unit. Detailed setup, configuration, and code belong in later topics and the module lab.

## Core Concepts

### Finding SageMaker in the AWS Console

Everything in AWS starts in the **AWS Management Console**, the browser-based dashboard you see after signing in at [console.aws.amazon.com](https://console.aws.amazon.com).

1. **Sign in** to the console. You land on the Console Home page, which shows recently visited services and a search bar at the top.
2. **Search for SageMaker.** Click the search bar (or press `Alt+S`) and type `SageMaker`. The dropdown shows **Amazon SageMaker** under Services. Click it.
3. **The SageMaker landing page** appears. On the left-hand navigation panel you will see sections like **Studio**, **Notebook instances**, **Training**, **Inference**, **Pipelines**, and **Governance**. Each section maps to a stage of the ML lifecycle. You do not need to memorize every link today; just notice that the sidebar organizes SageMaker's tools by *what you are trying to do* (build, train, deploy, govern).

> **Tip:** Pin SageMaker to your console navigation bar (click the star icon next to the service name) so it stays one click away during this skill unit.

### The core services at a glance

SageMaker is not a single tool. It is an umbrella over several services that work together. Here are the ones you will encounter most in this curriculum.

**SageMaker Studio** is the integrated development environment (IDE) for ML. It provides a browser-based workspace where you can write code in notebooks, manage experiments, launch training jobs, and review model artifacts without leaving one interface. In the console sidebar, Studio sits at the very top. You will configure it in the next topic (*Studio Domains & Profiles*), so for now just know it is the place where most of your hands-on work will happen.

**Amazon S3 (Simple Storage Service)** is where your data and model artifacts live. SageMaker reads training data from S3 buckets and writes trained model files (called *model artifacts*) back to S3. S3 is a separate AWS service with its own console page. To find it, search "S3" in the top search bar. You will see a list of buckets (containers for files). In practice, you create a bucket with a recognizable name like `sagemaker-<region>-<account-id>`, upload your dataset, and point SageMaker at that location. If you recall the *data ingestion* stage from the ML lifecycle reading, S3 is SageMaker's answer to "where does the data live in a controlled, versioned place?"

**AWS IAM (Identity and Access Management)** controls *who* and *what* can access your resources. Every time SageMaker launches a training job or creates an endpoint, it assumes an **IAM Execution Role** that grants it specific permissions (for example, reading from your S3 bucket or writing logs to CloudWatch). You can find IAM by searching "IAM" in the console search bar. Its dashboard shows users, roles, and policies. IAM has its own dedicated topic later in this module, so for now the key takeaway is: SageMaker never acts with your personal credentials; it uses a role you define with only the permissions it needs.

**SageMaker Training** is the managed training service. Instead of provisioning an EC2 instance, installing libraries, and babysitting a script, you describe the job (which script, which instance type, which data location) and SageMaker spins up the infrastructure, runs your code, saves the results to S3, and tears everything down. In the console sidebar, look under **Training > Training jobs** to see past and running jobs. This maps directly to the *model training* stage of the lifecycle.

**SageMaker Inference (Endpoints)** lets you deploy a trained model behind a managed HTTPS endpoint so applications can send data and receive predictions in real time. Under **Inference > Endpoints** in the sidebar, you can see active endpoints, their status, and the instance types powering them. This corresponds to the *deployment* stage. Later topics will walk through creating and invoking endpoints step by step.

### How the ecosystem maps to the ML lifecycle

If you completed the *ML Lifecycle & Reproducibility* reading in the Applied ML Foundations module, you saw the lifecycle as a loop: problem framing, data ingestion, EDA, feature engineering, training, evaluation, deployment, and monitoring. SageMaker provides a managed service for nearly every stage:

| ML Lifecycle Stage | SageMaker Equivalent | Where in the Console |
| :--- | :--- | :--- |
| Data ingestion & storage | Amazon S3 buckets | **S3** console (separate service) |
| EDA & experimentation | SageMaker Studio notebooks | **SageMaker > Studio** |
| Feature engineering | SageMaker Processing Jobs | **SageMaker > Processing > Processing jobs** |
| Training | SageMaker Training Jobs | **SageMaker > Training > Training jobs** |
| Evaluation | SageMaker Experiments / metrics | **SageMaker > Experiments** |
| Deployment | SageMaker Endpoints | **SageMaker > Inference > Endpoints** |
| Monitoring | SageMaker Model Monitor | **SageMaker > Inference > Model monitoring** |
| Governance & versioning | SageMaker Model Registry | **SageMaker > Governance > Model registry** |

You do not need to visit every one of these pages today. The table is a map: when you reach each lifecycle stage in later modules, you will know exactly which SageMaker service to reach for and where to find it in the sidebar.

### A note on SDK and CLI equivalents

Everything you can do in the console can also be done programmatically through the **SageMaker Python SDK** (`sagemaker` package) and the **AWS CLI** (`aws sagemaker ...`). In this curriculum, you will always learn the console workflow first so you understand what is happening behind the scenes, and then move to SDK code for automation and reproducibility. For now, it is enough to know the SDK exists; you will use it extensively starting in the Training module.

### Free Tier awareness

AWS offers a [Free Tier for SageMaker](https://aws.amazon.com/free/) that includes limited hours on specific instance types. Throughout this curriculum, exercises will use `ml.t3.medium` for notebooks and `ml.m5.xlarge` for training and inference to stay within those limits. Always check the *Billing & Cost Management* console (search "Billing" in the console search bar) to track your usage. Every deployment exercise will include explicit cleanup steps to delete endpoints and models, preventing unexpected charges.

## Connecting to Practice

This reading is your orientation to the SageMaker ecosystem. In the upcoming *Ecosystem & Core Services Video*, you will see a live walkthrough of the console navigation covered here. The next topic, *Studio Domains & Profiles*, will guide you through actually setting up your own SageMaker Studio environment. And in the module lecture and assignment, you will put these services to work end to end.

For now, the most valuable thing you can do is sign in to the AWS Console, search for SageMaker, and spend a few minutes clicking through the sidebar sections. Match each section to the lifecycle table above. Familiarity with the console layout will make every future lab smoother.

---

## Further Learning & Resources

**Documentation and reading**

- **[What Is Amazon SageMaker?](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)** - *Docs*: The official overview page covering every SageMaker capability, useful as a reference map when you encounter new features.
- **[AWS Free Tier - SageMaker](https://aws.amazon.com/sagemaker/pricing/)** - *Docs*: Detailed breakdown of free-tier allowances and pricing tiers so you can plan exercises without surprise bills.

**Interactive practice**

- **[AWS Hands-On Tutorials - Amazon SageMaker](https://aws.amazon.com/getting-started/hands-on/?getting-started-all.sort-by=item.additionalFields.content-latest-publish-date&getting-started-all.sort-order=desc&awsf.getting-started-category=category%23machine-learning&awsf.getting-started-content-type=content-type%23hands-on)** - *Interactive*: Free guided labs from AWS that walk you through SageMaker tasks in your own console, reinforcing the services introduced here.
- **[AWS Skill Builder - SageMaker Learning Plan](https://explore.skillbuilder.aws/learn/lp/2003/machine-learning-learning-plan)** - *Interactive*: Self-paced modules with quizzes that cover the SageMaker ecosystem from foundational to advanced levels.
