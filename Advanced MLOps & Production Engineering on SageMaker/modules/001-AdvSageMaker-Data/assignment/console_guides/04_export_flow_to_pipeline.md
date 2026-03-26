# Guide 4: Export Flow to a Pipeline

This guide shows you how to export your Data Wrangler flow into a SageMaker Pipeline processing step and execute the pipeline end to end. This converts your interactive data preparation work into a repeatable, automated workflow.

---

## Steps

### Step 1 -- Open the Data Wrangler Flow
1. In SageMaker Studio, open the Data Wrangler flow you saved in Guide 2.
2. Confirm all transform steps are present: Import, Impute, Encode, Scale.
3. Click the final transform node in the canvas to select it.

### Step 2 -- Export to a Pipeline
1. With the final node selected, click the **Export** tab (or click **Export to** at the top right).
2. From the export options, choose **SageMaker Pipeline (via Jupyter Notebook)**.
3. Data Wrangler generates a Jupyter notebook that defines a Pipeline with a Processing step.
4. The notebook opens automatically. Review the generated code to understand the pipeline definition.

### Step 3 -- Configure Pipeline Parameters
1. In the generated notebook, locate the cell that sets the **pipeline name**. Change it to `fraudshield-data-pipeline`.
2. Verify the **instance type** is set to `ml.m5.xlarge` and **instance count** is `1`.
3. Confirm the S3 output path points to your bucket: `s3://fraudshield-data-<account-id>/pipeline-output/`.
4. Verify the IAM role matches the execution role from Guide 1.

### Step 4 -- Run the Pipeline Notebook
1. Select the notebook kernel as **Python 3 (Data Science)** with an `ml.t3.medium` instance.
2. Run all cells in the notebook sequentially using **Run > Run All Cells**.
3. The final cell calls `pipeline.start()`. This submits the pipeline execution.
4. Note the pipeline execution ARN printed in the output.

### Step 5 -- Monitor the Pipeline in the Console
1. In the SageMaker console left navigation, expand **Pipelines** and click **Pipelines**.
2. Click `fraudshield-data-pipeline` to open it.
3. Click the most recent execution to view the execution graph.
4. Each step shows a status badge: **Executing**, **Succeeded**, or **Failed**.
5. Wait for the processing step to show **Succeeded** (this may take 5-10 minutes).

### Step 6 -- Verify Pipeline Output
1. Once the execution completes, click the processing step node in the graph.
2. In the step details panel, find the **Output** section and note the S3 path.
3. Navigate to S3 and open the output path. Verify that the transformed CSV or Parquet output files exist.
4. Download a sample file and spot-check that the transformations (imputation, encoding, scaling) have been applied.

---

## Presentation Checkpoint
Be prepared to show:
- The pipeline listed under **Pipelines** in the SageMaker console.
- The execution graph with the processing step in **Succeeded** status.
- The S3 output location containing the transformed dataset files.

---

## Key Concepts
- **SageMaker Pipeline:** A CI/CD-style workflow engine purpose-built for ML. Pipelines define a DAG of steps that can be versioned, scheduled, and audited.
- **Processing Step:** A pipeline step that runs a containerized data processing job, typically used for feature engineering, validation, or evaluation.
- **Pipeline Execution:** A single run of a pipeline. Each execution is tracked with a unique ARN, status, and lineage metadata.
- **Idempotent Export:** Data Wrangler encodes every transformation in the exported notebook, ensuring the pipeline reproduces exactly the same output as the interactive flow.
