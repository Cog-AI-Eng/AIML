# BYOM & BYOS Approaches

**Estimated Time:** 10 Minutes

## Introduction

In Module 1 you deployed a pre-built model from JumpStart without writing any training code. That shortcut works when a general-purpose model fits your problem. But in the Applied ML Foundations module, you wrote your own scikit-learn training scripts -- choosing algorithms, engineering features, tuning hyperparameters, and evaluating results on your own data. When you need that level of control on SageMaker's managed infrastructure, you need a way to bring your own code into the platform.

SageMaker offers two paths for this: **Bring Your Own Script (BYOS)**, commonly called **Script Mode**, and **Bring Your Own Model/Container (BYOM)**. Think of it as the difference between renting a furnished apartment and building a house from the ground up. Script Mode gives you a pre-built environment (a managed Docker container with frameworks like scikit-learn, PyTorch, or TensorFlow already installed) and lets you supply just your training script. BYOM gives you a blank lot and lets you construct the entire container yourself, including the operating system, dependencies, and entry points.

For most projects -- and for this entire curriculum -- **Script Mode is the recommended approach**. It offers the right balance of control and convenience. BYOM with custom containers is the advanced path for teams with specialized dependency requirements that no managed container supports. This reading explains both approaches, shows where each option appears in the SageMaker console, and helps you decide which to use.

## Core Concepts

### The customization spectrum

SageMaker's training options form a spectrum from least to most customization:

| Approach | What You Provide | What SageMaker Provides | Best For |
| :--- | :--- | :--- | :--- |
| **Built-in algorithms** | Data and hyperparameters only | Algorithm, container, training logic | Standard problems with no custom code |
| **Script Mode (BYOS)** | Your training script (`.py` file) | Managed container with framework pre-installed | Custom training logic with standard frameworks |
| **Bring Your Own Container (BYOM)** | Complete Docker image with all code and dependencies | Compute infrastructure only | Specialized dependencies or proprietary frameworks |

JumpStart (Module 1) sits to the left of this spectrum -- it even handles data preparation in some cases. Script Mode sits in the middle, where you get managed infrastructure but full control over your training logic. BYOM sits at the far right, where you control everything inside the container.

### Script Mode (BYOS) -- the recommended path

Script Mode is the approach you will use throughout this curriculum. The idea is straightforward: SageMaker provides a pre-built Docker container for your chosen framework (scikit-learn, PyTorch, TensorFlow, XGBoost, and others), and you supply a Python script that contains your training logic. SageMaker injects your script into the container, runs it on managed compute, and saves the output to S3.

If you have written a training script in the AIML Foundations module -- loading data, fitting a model, evaluating metrics, saving the result -- you already know how to write a Script Mode entry point. The main differences are:

- **Data comes from S3, not a local file path.** SageMaker downloads your training data from S3 into a known directory inside the container before your script runs. Your script reads from that directory instead of a local path.
- **Model output goes to a specific directory.** Instead of saving a model to any path you choose, you save it to a designated output directory. SageMaker compresses that directory into a `model.tar.gz` artifact and uploads it to S3 after training completes.
- **Hyperparameters arrive as command-line arguments.** Instead of hardcoding values, your script accepts hyperparameters through an argument parser. SageMaker passes them automatically.

The next topic, *Script Mode Structure*, covers the exact script format in detail. For now, the key takeaway is that Script Mode lets you reuse familiar Python training code with minimal changes.

### Where Script Mode appears in the console

When you create a training job through the SageMaker console, the Script Mode option appears during the algorithm selection step. Here is how to find it:

1. **Navigate to Training Jobs.** In the SageMaker console sidebar, click **Training > Training jobs**, then click **Create training job**.

2. **Algorithm source.** On the training job creation form, you will see an **Algorithm source** section with options:
   - **SageMaker built-in algorithm** -- selects one of SageMaker's pre-packaged algorithms (XGBoost, Linear Learner, etc.) with no custom code.
   - **Your own algorithm container in ECR** -- this is the BYOM path (covered below).
   - **An algorithm from AWS Marketplace** -- third-party algorithms.

   For Script Mode, you select **SageMaker built-in algorithm** but choose a *framework* image (like "scikit-learn" or "PyTorch") rather than a specific algorithm. The framework image is the managed container that will run your script.

3. **Entry point and source directory.** After selecting the framework, the form shows fields for:
   - **Entry point** -- the filename of your training script (e.g., `train.py`).
   - **Source directory** -- the S3 location of your script (or a local path if using Studio).

   These fields tell SageMaker which script to inject into the container and run.

4. **Continue configuring** the rest of the training job (instance type, data channels, output location). The *Training Job Anatomy* topic later in this module covers these fields in detail.

> **Tip:** Most developers configure Script Mode training jobs through the SDK rather than the console form, because it is easier to iterate on code. However, understanding the console form helps you see exactly what parameters a training job requires -- every SDK parameter maps to a field on this form.

### Bring Your Own Container (BYOM) -- the advanced path

BYOM is for situations where the managed containers do not have what you need. Common reasons include:

- Your training code depends on a library that is not available in any SageMaker-managed container.
- You need a specific version of CUDA, a custom C++ extension, or a proprietary framework.
- Your organization has compliance requirements that mandate a specific base image.

With BYOM, you build a complete Docker image, push it to **Amazon Elastic Container Registry (ECR)**, and tell SageMaker to use it. Your image must follow SageMaker's container contract: it must read input data from specific directories, write output to specific directories, and respond to specific environment variables. SageMaker does not inject any framework into your image -- you are responsible for everything inside it.

### Where BYOM appears in the console

On the same training job creation form described above, the BYOM option appears as:

1. **Algorithm source > Your own algorithm container in ECR.** Selecting this option reveals a field for the **ECR image URI** -- the full path to your Docker image in ECR (e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-training-image:latest`).

2. **No entry point field.** Unlike Script Mode, there is no entry point or source directory field. The container itself defines what runs when SageMaker starts the training job (typically through a `ENTRYPOINT` or `CMD` in the Dockerfile).

3. The rest of the form (instance type, data channels, output path) is identical to Script Mode.

### Deciding between Script Mode and BYOM

For this curriculum, the answer is always Script Mode. But here is the decision framework for real-world projects:

| Question | If Yes | If No |
| :--- | :--- | :--- |
| Does a SageMaker-managed container exist for your framework (scikit-learn, PyTorch, TensorFlow, XGBoost)? | Use Script Mode | Consider BYOM |
| Do you need libraries not available in the managed container? | Consider BYOM or extending the managed container | Use Script Mode |
| Do you need full control over the OS, CUDA version, or system libraries? | Use BYOM | Use Script Mode |
| Are you learning SageMaker for the first time? | Use Script Mode | Use Script Mode |

There is also a middle ground: **extending a managed container**. You can take a SageMaker-provided container image, add your custom libraries on top via a Dockerfile, push the extended image to ECR, and use it with the BYOM path. This gives you the framework baseline plus your custom dependencies. However, this approach is beyond the scope of this curriculum.

### SDK equivalents

In the SDK, the distinction between Script Mode and BYOM shows up in how you configure the `Estimator` object:

- **Script Mode:** You use a framework-specific estimator (e.g., `sagemaker.sklearn.SKLearn`, `sagemaker.pytorch.PyTorch`) and pass your script via the `entry_point` parameter. The estimator knows which managed container to use.
- **BYOM:** You use the generic `sagemaker.estimator.Estimator` class and pass your ECR image URI via the `image_uri` parameter. No entry point is needed because the container defines its own.

You will work with framework estimators starting in the *Estimators & Configurations* topic later in this module.

## Connecting to Practice

This reading sets up the conceptual framework for the rest of Module 2. Every subsequent topic in this module builds on Script Mode: the next reading (*Script Mode Structure*) shows you the exact format your training script must follow, *Estimators & Configurations* shows you how to configure the SDK to run it, *Training Job Anatomy* shows you what happens behind the scenes, and *Model Artifacts & S3 Storage* shows you where the results end up.

The most useful thing you can do right now is revisit a training script you wrote in the AIML Foundations module (loading data, fitting a model, saving results) and think about what would need to change to run it on SageMaker. Where does the data come from? Where does the model file go? How do hyperparameters get passed in? The next reading answers all three questions.

---

## Further Learning & Resources

**Documentation and reading**

- **[Use Your Own Training Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html)** - *Docs*: The official reference for both Script Mode and BYOM, including the container contract that custom images must follow.
- **[Pre-built SageMaker Docker Images](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-frameworks.html)** - *Docs*: A complete list of managed framework containers available for Script Mode, with supported framework versions.

**Interactive practice**

- **[AWS Hands-On: Train a Model with Script Mode](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-train-a-model/)** - *Interactive*: A free guided lab walking through a Script Mode training job from notebook to S3 artifact.
- **[SageMaker Examples - Script Mode](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-script-mode)** - *Interactive*: A collection of runnable notebook examples demonstrating Script Mode with scikit-learn, PyTorch, and TensorFlow.
