# Guide 5: Deploy a JumpStart Pre-built Model

SageMaker JumpStart provides a catalog of pre-trained models that you can deploy with a few clicks. This is the fastest way to get a live inference endpoint running and validates that your Studio environment is working correctly before you start training custom models in later modules.

---

## Steps

### Step 1 -- Open JumpStart in Studio

1. From the **SageMaker console**, go to **Domains** -> **fraudshield-domain**.
2. Click **Open Studio** next to your default user profile.
3. In Studio, click the **SageMaker Home** icon (house icon) in the left sidebar.
4. Click **JumpStart** to open the model catalog.

### Step 2 -- Browse the Model Catalog

1. Take a moment to explore the catalog. You will see categories like:
   - **Foundation Models** (large language models)
   - **Vision** (image classification, object detection)
   - **Text** (text classification, sentiment analysis)
   - **Tabular** (classification, regression on structured data)
2. For this lab, we need a **lightweight model** that deploys quickly and uses a Free Tier-eligible instance.
3. In the search bar, search for a text classification or tabular model. Look for models that list `ml.m5.xlarge` as a supported instance type.
4. Select a model. Good options include lightweight text classification or tabular models -- avoid large foundation models as they require expensive GPU instances.

### Step 3 -- Review the Model Details

Before deploying, review the model page:

1. **Model description:** What task does this model perform?
2. **Supported instance types:** Confirm `ml.m5.xlarge` is listed.
3. **Input/output format:** What data format does the model expect?
4. **License:** Note any usage restrictions.

### Step 4 -- Deploy the Model

1. Click **Deploy**.
2. In the deployment configuration:
   - **Endpoint name:** Enter `fraudshield-demo-endpoint`
   - **Instance type:** Select `ml.m5.xlarge` (Free Tier eligible for the first 2 months)
   - **Instance count:** `1`
3. Leave other settings as default.
4. Click **Deploy**.

### Step 5 -- Wait for the Endpoint to Become Active

1. Deployment typically takes 5-10 minutes. You will see a progress indicator.
2. While waiting, navigate to the **SageMaker console** in a separate tab.
3. Go to **Inference** -> **Endpoints** in the left navigation.
4. Find `fraudshield-demo-endpoint` in the list. The status will show **Creating**.
5. Refresh periodically until the status changes to **InService**.

### Step 6 -- Explore the Endpoint Details

1. Click on `fraudshield-demo-endpoint` to open the details page.
2. Note the following:
   - **Endpoint status:** InService
   - **Endpoint configuration name:** (auto-generated)
   - **Creation time**
   - **Instance type** and **count** in the production variants section
3. Navigate to **Inference** -> **Endpoint configurations** and find the configuration associated with your endpoint. Note:
   - The **Model name** it references
   - The **variant weight** (should be 1.0)
4. Navigate to **Inference** -> **Models** and find the model object. Note:
   - The **container image** (ECR URI)
   - The **model data URL** (S3 path to the model artifacts)

### Step 7 -- Observe the Three-Object Pattern

You have just created three objects that work together. This is the **three-object deployment pattern** you will use extensively in Module 3:

1. **Model** -- Points to the model artifacts in S3 and the inference container image
2. **Endpoint Configuration** -- Defines the instance type, count, and which model to deploy
3. **Endpoint** -- The live service that accepts prediction requests

All three appear in the SageMaker console under **Inference**. Understanding this pattern now will make Module 3's deployment lab much clearer.

### Step 8 -- Test the Endpoint (Optional)

If the JumpStart model provided a sample notebook:

1. Return to Studio and look for an **Open Notebook** option on the model page.
2. Run the sample invocation cells to send a test prediction to your endpoint.
3. Observe the response format.

If no sample notebook is available, that is fine -- you will learn to invoke endpoints programmatically in Module 3.

---

## Presentation Checkpoint

Be prepared to show:
- The `fraudshield-demo-endpoint` in the **Inference -> Endpoints** list with **InService** status
- The associated **Endpoint Configuration** and **Model** objects in their respective console sections
- The three-object relationship: which model does the config reference, and which config does the endpoint use?
- If you ran a sample prediction, show the request and response
- Explain: Why does SageMaker separate the Model, Endpoint Configuration, and Endpoint into three objects instead of combining them? (Flexibility -- you can update the model without recreating the endpoint, run multiple model versions behind the same endpoint, or change instance types via a new configuration)

---

## Important Cost Warning

Your endpoint is now **running and incurring charges**. `ml.m5.xlarge` endpoints cost approximately $0.23/hour. You MUST complete Guide 6 (Cleanup) before stepping away from this lab. Do not leave the endpoint running overnight.

---

## Key Concepts

- **JumpStart:** A curated catalog of pre-trained models, solutions, and example notebooks. Enables rapid experimentation without building models from scratch.
- **Inference Endpoint:** A live HTTPS service hosted by SageMaker that accepts prediction requests and returns results. Backed by EC2 instances running your model inside a Docker container.
- **Three-Object Pattern:** Model (what to run) + Endpoint Configuration (how to run it) + Endpoint (where to run it). This separation of concerns is fundamental to SageMaker deployments.

---

## AIML Connection

The *Algorithm Selection Framework* reading discussed choosing the right model for a task. JumpStart gives you access to dozens of pre-built models for common tasks (classification, regression, NLP, vision). In practice, teams often start with a JumpStart model as a baseline and then train custom models only if the pre-built option does not meet their performance requirements.
