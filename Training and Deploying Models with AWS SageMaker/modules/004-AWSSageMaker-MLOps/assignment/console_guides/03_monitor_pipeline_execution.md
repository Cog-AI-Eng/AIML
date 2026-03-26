# Guide 3: Monitor Pipeline Execution in the Console

The SageMaker console provides a visual DAG view of your pipeline execution, showing each step's status in real time. This is the primary way to monitor, debug, and understand pipeline runs.

---

## Steps

### Step 1 -- Find Your Pipeline

1. In the **SageMaker console**, go to **Pipelines** -> **Pipelines** in the left navigation.
2. You should see `fraudshield-detection-pipeline` listed.
3. Click on it to open the pipeline details.

### Step 2 -- Explore the Pipeline Details

On the pipeline page, you will see several tabs:

| Tab | What It Shows |
|-----|--------------|
| **Graph** | The DAG visualization of the pipeline steps |
| **Parameters** | The defined parameters and their default values |
| **Executions** | All past and current executions |
| **Settings** | Pipeline configuration and role |

Click on the **Graph** tab to see the DAG:
- **PreprocessData** -> **TrainModel** -> **RegisterModel**
- The arrows show dependencies: TrainModel waits for PreprocessData, RegisterModel waits for TrainModel.

### Step 3 -- Monitor the Active Execution

1. Click on the **Executions** tab.
2. Find the execution you started in Guide 2. It will show a status:
   - **Executing** (blue) -- currently running
   - **Succeeded** (green) -- all steps completed
   - **Failed** (red) -- a step failed
3. Click on the execution to open its details.

### Step 4 -- Watch the DAG in Real Time

1. On the execution details page, you will see the DAG with color-coded steps:

| Color | Meaning |
|-------|---------|
| **Gray** | Not started (waiting for dependencies) |
| **Blue** | Currently executing |
| **Green** | Completed successfully |
| **Red** | Failed |
| **Yellow** | Stopping |

2. Watch the steps progress:
   - First, **PreprocessData** turns blue (executing), then green (completed)
   - Then **TrainModel** turns blue, then green
   - Finally **RegisterModel** turns blue, then green

3. Refresh the page periodically to see updates. A full execution typically takes 15-25 minutes.

### Step 5 -- Explore Step Details

Click on each step in the DAG to see its details:

**PreprocessData:**
- Input and output S3 paths
- Processing job name
- Logs (link to CloudWatch)
- Duration

**TrainModel:**
- Training job name
- Hyperparameters used
- Model artifact location
- CloudWatch logs

**RegisterModel:**
- Model package version created
- Registry group name
- Approval status (`PendingManualApproval`)

### Step 6 -- Verify the Registered Model

1. Go to **Governance** -> **Model registry** -> `fraud-detection-rf`.
2. You should see a new version registered by the pipeline (e.g., Version 2 if Version 1 was from Module 3).
3. Note that the status is `PendingManualApproval` -- the pipeline registered the model but did not approve it. Approval is still a human decision (or an automated check, which you will add in Guide 4).

### Step 7 -- Check S3 for Pipeline Outputs

1. Navigate to **S3** -> your bucket.
2. Look for the `pipeline/` prefix:
   - `pipeline/processed/train/` -- preprocessed training data
   - `pipeline/processed/test/` -- preprocessed test data
   - `pipeline/output/<job-name>/output/model.tar.gz` -- the trained model artifact
3. This shows how the pipeline steps create a chain of artifacts in S3.

### Step 8 -- Review Pipeline Parameters

1. Return to the pipeline details and click the **Parameters** tab.
2. Note the parameters you defined:
   - `TrainingInstanceType`: `ml.m5.xlarge`
   - `NEstimators`: `100`
3. When starting a new execution, you could override these values -- for example, `NEstimators=200` for a different experiment -- without changing the pipeline code.

---

## Presentation Checkpoint

Be prepared to show:
- The **DAG visualization** with all steps showing green (succeeded)
- The step details for at least one step (inputs, outputs, logs)
- The new model version in the **Model Registry** created by the pipeline
- The pipeline outputs in **S3** (`pipeline/processed/`, `pipeline/output/`)
- Explain: What does each color mean in the DAG visualization? (Gray = waiting, Blue = executing, Green = succeeded, Red = failed)
- Explain: How does the pipeline know to run TrainModel after PreprocessData? (The TrainingStep references the ProcessingStep's output properties, creating an implicit dependency. SageMaker reads these references and builds the execution order.)
- Explain: What is the advantage of pipeline parameters over hardcoded values? (Parameters make the pipeline reusable -- you can run different experiments with different hyperparameters, instance types, or data paths without editing the code)

---

## Key Concepts

- **DAG Visualization:** The console renders the pipeline as a directed graph where nodes are steps and edges are dependencies. This makes it easy to understand the workflow at a glance.
- **Execution:** A single run of a pipeline with specific parameter values. Each execution is independent and creates its own set of artifacts.
- **Step Properties:** The mechanism that connects steps. When step B references step A's output, SageMaker automatically ensures A runs before B.
- **Implicit Dependencies:** You never explicitly say "run B after A." Instead, you reference A's output in B's definition, and SageMaker infers the dependency.
