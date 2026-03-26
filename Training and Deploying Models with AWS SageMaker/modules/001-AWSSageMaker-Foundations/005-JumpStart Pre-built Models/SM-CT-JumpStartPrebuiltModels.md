# JumpStart Pre-built Models

**Estimated Time:** 10 Minutes

## Introduction

Over the previous four readings you built a foundation: you mapped the SageMaker ecosystem, set up a Studio Domain, secured permissions with IAM, and learned the five-stage ML lifecycle (Prepare, Build, Train & Tune, Deploy, Monitor). That lifecycle assumes you will write your own training scripts, choose your own algorithms, and run your own training jobs. For many production projects, that is exactly what you will do. But sometimes the fastest path to a working model is to skip the Build and Train stages entirely and start with something that already works.

That is what **SageMaker JumpStart** provides. JumpStart is a curated library of pre-trained models and solution templates that you can browse, evaluate, and deploy directly from the SageMaker console -- no custom training code required. If you have completed the *Algorithm Selection Framework* reading in the Applied ML Foundations module, you already know the decision process for choosing between regression, classification, clustering, and other problem types. JumpStart lets you act on that decision immediately: pick a problem type, find a model that fits, deploy it, and start generating predictions.

This reading walks you through the JumpStart experience in the AWS Management Console, shows you how to deploy a pre-built model to an endpoint, and includes the mandatory cleanup steps to avoid unnecessary billing.

## Core Concepts

### What JumpStart offers

JumpStart is organized around three categories of content:

**Pre-trained models** are models that have already been trained on large datasets and are ready to use for inference. These include foundation models for text generation, image classification models, object detection models, and tabular prediction models. You deploy them as-is and send data to get predictions.

**Fine-tunable models** are pre-trained models that you can further train on your own dataset to improve performance for your specific use case. Fine-tuning is a middle ground between using a pre-built model and training from scratch. JumpStart provides the fine-tuning scripts and infrastructure; you supply the data.

**Solution templates** are end-to-end example architectures for common ML use cases (fraud detection, demand forecasting, document understanding). Each template includes notebooks, training scripts, and deployment configurations. These are useful for learning patterns but are not the focus of this reading.

For this curriculum, the primary focus is on deploying pre-trained models -- the fastest path from zero to a working endpoint.

### Finding JumpStart in the console

JumpStart lives inside SageMaker Studio. You access it through the browser, not through a separate console page.

1. **Open SageMaker Studio.** In the SageMaker console sidebar, navigate to your Domain and launch Studio for your User Profile (as you learned in the *Studio Domains & Profiles* reading).

2. **Navigate to JumpStart.** Once Studio loads, look at the left sidebar inside the Studio interface. Click on the **JumpStart** icon (it looks like a rocket or may be labeled "JumpStart" depending on your Studio version). Alternatively, you can find JumpStart under the **Home** landing page in Studio, where it appears as a prominent section.

3. **Browse the model hub.** The JumpStart page displays a searchable catalog of models organized by task type:
   - **Text generation** -- large language models for summarization, question answering, and content generation.
   - **Image classification** -- models that label images into categories.
   - **Object detection** -- models that identify and locate objects within images.
   - **Tabular classification and regression** -- models for structured data predictions (the type you explored with scikit-learn in the AIML Foundations module).
   - **Sentence embeddings** -- models that convert text into numerical vectors for similarity search.

   You can filter by provider (AWS, Hugging Face, Meta, Stability AI), by task, or by keyword search.

### Deploying a pre-built model step by step

Here is the console walkthrough for deploying a model from JumpStart. This example uses a text classification model, but the steps are the same for any model type.

1. **Select a model.** Browse or search the JumpStart catalog. Click on a model card to open its detail page. The detail page shows:
   - A description of what the model does.
   - The model provider and license.
   - The default instance type for deployment.
   - Sample input/output formats.
   - A **Deploy** button.

2. **Review the deployment configuration.** Before clicking Deploy, review the configuration panel:
   - **Endpoint name:** A default name is generated. You can change it to something descriptive like `jumpstart-text-classify-demo`.
   - **Instance type:** JumpStart pre-selects an instance type based on the model's requirements. For this curriculum, verify that the instance type is `ml.m5.xlarge` or smaller to stay within Free Tier limits. If the default is a GPU instance (like `ml.p3.2xlarge`), change it to `ml.m5.xlarge` unless the model specifically requires GPU. Some lighter models also support **Serverless Inference** configuration, which scales to zero when idle and minimizes costs.
   - **Security settings:** The deployment uses your User Profile's execution role. The role you configured in the *IAM* reading determines what this endpoint can access.

3. **Click Deploy.** SageMaker provisions the infrastructure, loads the model, and creates an endpoint. This process takes several minutes. The JumpStart UI shows a progress indicator with status updates.

4. **Verify the endpoint.** Once deployment completes, the status changes to **InService**. You can verify this in two places:
   - On the JumpStart model detail page, which shows the endpoint status.
   - In the SageMaker console sidebar under **Inference > Endpoints**, where your new endpoint appears in the list.

5. **Test the endpoint.** Many JumpStart model pages include an **Open Notebook** option that generates a sample notebook with code to invoke the endpoint. Click it to open a pre-built notebook in Studio. The notebook contains sample input data and a `predict()` call so you can verify the model is working. You will learn to write your own invocation code in the *Invoking Endpoints* topic in Module 3.

### Mandatory cleanup

Every deployed endpoint incurs charges for every minute it runs, even if no one is sending requests. You must delete the endpoint and model when you are done testing. This is a strict requirement throughout this curriculum.

**Cleanup via the console:**

1. Navigate to **SageMaker > Inference > Endpoints** in the console sidebar.
2. Select the checkbox next to your endpoint.
3. Click **Actions > Delete**. Confirm the deletion.
4. Navigate to **SageMaker > Inference > Models**.
5. Select the checkbox next to the model associated with your endpoint.
6. Click **Actions > Delete**. Confirm the deletion.

**Cleanup via code (for reference):**

```python
predictor.delete_endpoint()
predictor.delete_model()
```

Both approaches achieve the same result. Use whichever you are more comfortable with, but always verify in the console that the endpoint and model are gone. An endpoint that shows **InService** is costing money.

> **Critical reminder:** Check your *Billing & Cost Management* console (search "Billing" in the console search bar) after cleanup to confirm no unexpected charges. Make this a habit after every lab and exercise.

### When to use JumpStart vs. custom training

JumpStart is powerful for rapid prototyping and for problems where a general-purpose model is sufficient. But it is not always the right choice:

| Scenario | Recommended Approach |
| :--- | :--- |
| You need a quick proof-of-concept for a standard task (text classification, image labeling) | JumpStart pre-trained model |
| You have a standard task but need higher accuracy on your specific data | JumpStart fine-tunable model |
| You have a unique problem, custom data formats, or specialized domain requirements | Custom training (Module 2) |
| You want to understand the full training pipeline for learning purposes | Custom training (Module 2) |

If you recall the *Algorithm Selection Framework* from the AIML module, JumpStart is essentially a shortcut past the "which algorithm should I pick?" question. The catalog does the selection for you based on your task type. Custom training gives you full control when the shortcut does not fit.

### SDK equivalent

After you are comfortable deploying from the console, you can deploy JumpStart models programmatically using the SageMaker Python SDK. The `sagemaker.jumpstart.model.JumpStartModel` class lets you specify a model ID, deploy it to an endpoint, and invoke it -- all in a few lines of Python. You will see SDK-based deployments in later modules. The console workflow ensures you understand what each step does before automating it.

## Connecting to Practice

This reading completes the Foundations module. You now have the full picture: the ecosystem, your Studio workspace, your IAM permissions, the five-stage lifecycle, and a shortcut (JumpStart) for rapid deployment. In the *JumpStart Pre-built Models Video*, you will see a recorded walkthrough of the deployment and cleanup steps. In the module lecture, you will deploy a JumpStart model yourself and test it end to end. And in the module assignment, you will apply these concepts hands-on.

The most useful thing you can do right now is open JumpStart in Studio and spend a few minutes browsing the catalog. Click into two or three model cards and read their descriptions. Notice the instance type defaults and think about which ones fit within Free Tier limits. Do not deploy anything yet unless you are prepared to follow the cleanup steps immediately afterward.

---

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)** - *Docs*: The official documentation covering model discovery, deployment options, and fine-tuning workflows within JumpStart.
- **[SageMaker JumpStart Available Models](https://aws.amazon.com/sagemaker/jumpstart/getting-started/)** - *Docs*: A browsable catalog of available JumpStart models with filtering by task type and provider, useful for planning which model to deploy.

**Interactive practice**

- **[AWS Hands-On: Deploy a Model with Amazon SageMaker JumpStart](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-deploy-model-to-real-time-inference-endpoint/)** - *Interactive*: A free guided lab that walks you through deploying and testing a JumpStart model in your own console.
- **[AWS Skill Builder - Exploring SageMaker JumpStart](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/17432/exploring-amazon-sagemaker-jumpstart)** - *Interactive*: Self-paced digital training with hands-on exercises covering JumpStart model discovery and deployment.
