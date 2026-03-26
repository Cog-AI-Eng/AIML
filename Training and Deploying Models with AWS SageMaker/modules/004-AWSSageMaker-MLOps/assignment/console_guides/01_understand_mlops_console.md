# Guide 1: Understand MLOps in the Console

Before building a pipeline, take a tour of the SageMaker console sections that support MLOps workflows. Understanding where each tool lives in the console helps you see how the pieces connect.

---

## Steps

### Step 1 -- Review the MLOps Concept

MLOps applies DevOps principles to machine learning:

| DevOps Concept | MLOps Equivalent |
|---------------|-----------------|
| Source code | Training code + training data (two sources of change) |
| CI (Continuous Integration) | Continuous Training -- retrain when data or code changes |
| CD (Continuous Deployment) | Continuous Deployment -- promote approved models to endpoints |
| Monitoring | Model Monitor -- detect data drift and prediction degradation |
| Pipelines (CI/CD) | SageMaker Pipelines -- orchestrate the ML workflow as code |

The key difference from traditional DevOps: ML systems have **two sources of change** (code AND data), so the automation must handle both.

### Step 2 -- Tour the Pipelines Section

1. In the **SageMaker console**, go to **Pipelines** -> **Pipelines** in the left navigation.
2. This section will likely be empty (you have not created a pipeline yet).
3. When you create a pipeline in Guide 2, this is where you will see:
   - Pipeline definitions (the DAG structure)
   - Pipeline executions (individual runs)
   - Execution status and step details

### Step 3 -- Revisit the Model Registry

1. Go to **Governance** -> **Model registry**.
2. Click on `fraud-detection-rf` (created in Module 3).
3. Note how Version 1 is already registered and approved.
4. In Guide 2, the pipeline will automatically register new versions here, creating a continuous flow from training to governance.

### Step 4 -- Explore the Model Monitor Section

1. Go to **Inference** -> **Model monitoring** (or look for a "Monitor" section in the left navigation).
2. This section may be empty or show monitoring-related options.
3. Model Monitor would be configured here to:
   - Capture prediction inputs and outputs from a live endpoint
   - Compare incoming data against a training data baseline
   - Detect **data drift** (input features shifting) and **concept drift** (relationship between features and target changing)
   - Send alerts when drift is detected
4. You will NOT configure Model Monitor in this lab -- this is a conceptual tour. Guide 5 covers this in more detail.

### Step 5 -- Explore EventBridge (Briefly)

1. Navigate to **Amazon EventBridge** (search for "EventBridge" in the top search bar).
2. Click **Rules** in the left navigation.
3. EventBridge is AWS's event bus -- it can trigger actions when specific events happen. For MLOps:
   - "When a model is registered in the Registry" -> trigger a notification
   - "When Model Monitor detects drift" -> trigger pipeline retraining
   - "When a pipeline execution completes" -> trigger deployment
4. You will NOT create rules in this lab -- Guide 5 covers the concept.

### Step 6 -- Map Console Sections to the MLOps Lifecycle

Document how the console sections map to the MLOps lifecycle stages:

| MLOps Stage | Console Section | What It Does |
|-------------|----------------|-------------|
| **Prepare** | S3 + Processing Jobs | Store and preprocess training data |
| **Build & Train** | Training Jobs + Pipelines | Train models on managed compute |
| **Evaluate** | Processing Jobs (evaluation scripts) | Compute metrics on test data |
| **Register** | Model Registry | Version and govern model artifacts |
| **Approve** | Model Registry (approval status) | Quality gate before deployment |
| **Deploy** | Inference (Model, Config, Endpoint) | Serve predictions in production |
| **Monitor** | Model Monitoring | Detect drift and degradation |
| **Automate** | Pipelines + EventBridge | Orchestrate and trigger workflows |

---

## Presentation Checkpoint

Be prepared to show:
- The console sections for Pipelines, Model Registry, Model Monitoring, and EventBridge
- Your mapping of console sections to MLOps lifecycle stages
- Explain: What are the two sources of change in ML systems that make MLOps different from traditional DevOps? (Code changes AND data changes. A model may need retraining not because the code changed but because the incoming data distribution shifted.)
- Explain: What is the difference between data drift and concept drift? (Data drift: input feature distributions change. Concept drift: the relationship between features and the target variable changes. Both degrade model performance.)
- Explain: How would EventBridge connect Model Monitor to SageMaker Pipelines? (Monitor detects drift -> EventBridge rule matches the drift event -> triggers a Pipeline execution to retrain the model on fresh data)

---

## Key Concepts

- **MLOps Maturity Levels:** Level 0 (manual everything), Level 1 (automated training pipeline), Level 2 (automated training + deployment + monitoring). This lab covers Level 1.
- **Continuous Training:** Automatically retraining models when data changes, triggered by schedules or drift detection.
- **Data Drift:** A shift in the distribution of input features compared to the training data baseline.
- **Concept Drift:** A change in the relationship between features and the target variable, making the model's learned patterns obsolete.
