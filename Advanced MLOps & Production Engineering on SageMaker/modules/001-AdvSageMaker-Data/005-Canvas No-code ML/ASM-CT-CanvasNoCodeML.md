# Canvas No-code ML

**Estimated Time:** 10 Minutes

## Introduction

Not every team member who benefits from ML predictions knows how to write Python. Business analysts, product managers, and domain experts often have the closest relationship with the data and the clearest understanding of what predictions would be valuable, but they lack the engineering skills to build models themselves. SageMaker Canvas bridges this gap by providing a fully visual, no-code interface for building, evaluating, and deploying ML models directly from the browser.

Canvas runs within a SageMaker Studio Domain, meaning it shares the same IAM governance, VPC isolation, and cost controls you configured in the *Studio & Classic Domains* topic. But its user experience is fundamentally different from Studio notebooks: there is no code editor, no terminal, and no need to understand scikit-learn or pandas. Users upload or connect to data, select a target column, click "Build," and Canvas handles algorithm selection, hyperparameter tuning, feature engineering, and model evaluation automatically.

This reading covers how Canvas works architecturally, how to launch and use it from the console, and where it fits in an enterprise ML strategy alongside code-first workflows.

## Core Concepts

### Launching Canvas

Canvas is accessed through Studio. To open it:

1. Navigate to **SageMaker > Domains > [your domain]** in the console.
2. Find the user profile you want to use and click **Launch > Canvas**.
3. Canvas opens in a new browser tab with its own interface (separate from the JupyterLab IDE).

Canvas provisions a dedicated compute instance for the session. This instance runs as long as the Canvas tab is active. Always close Canvas when not in use to avoid charges.

### Data import in Canvas

Canvas supports multiple data sources:

- **Local upload:** Drag and drop CSV files up to 5 GB.
- **Amazon S3:** Browse buckets and select files directly.
- **Amazon Redshift, Amazon Athena:** Connect to data warehouses and query tables.
- **Snowflake:** Native connector for Snowflake databases.
- **Third-party SaaS:** Connectors for Salesforce, SAP, Google Analytics, and others (availability varies by region).

After importing, Canvas shows a spreadsheet-like view of the data with column-level statistics (type, missing values, unique values). You can filter, sort, rename columns, and join multiple datasets visually before building.

### Building a model

1. **Select a dataset** from your imported data.
2. **Choose the target column** -- the column you want to predict.
3. Canvas automatically detects the problem type:
   - **Binary classification** (two-class target)
   - **Multi-class classification** (three or more classes)
   - **Regression** (numeric target)
   - **Time series forecasting** (if you specify a time column and item identifier)
4. **Choose build mode:**
   - **Quick Build:** Trains a model in 2-15 minutes using a random sample and limited algorithm space. Good for rapid validation of whether the data supports the prediction task.
   - **Standard Build:** Runs a full Autopilot job under the hood (see next topic) with comprehensive algorithm search and hyperparameter tuning. Takes 2-4 hours depending on data size.
5. Click **Build**. Canvas shows a progress screen and notifies you when training is complete.

### Evaluating results

After a build completes, Canvas presents an evaluation dashboard:

- **Accuracy metrics:** For classification, it shows overall accuracy, per-class precision, recall, and F1. For regression, it shows RMSE and MAE.
- **Column impact:** A ranked list of features by their influence on the prediction (similar to SHAP feature importance but presented without requiring the user to understand SHAP).
- **Advanced metrics:** Confusion matrix (classification) or residual distribution (regression).
- **What-if analysis:** An interactive tool where you set feature values and see the predicted outcome in real time. This is particularly valuable for business users who want to understand "what would happen if..." scenarios.

### Generating predictions

Canvas offers two prediction modes:

- **Single prediction:** Use the what-if analysis panel to input values and see a prediction immediately.
- **Batch prediction:** Upload a CSV of new data (without the target column) and Canvas returns a CSV with predictions appended. This runs as a batch transform job.

### Sharing and collaboration

Canvas models exist within the user's profile in the Domain. To share a model with a data scientist for further refinement:

1. In Canvas, select the model and click **Share**.
2. Choose a target user profile within the same Domain.
3. The recipient can open the shared model in Studio notebooks as an Autopilot model artifact, inspect the generated code, and fine-tune it programmatically.

This sharing workflow is the recommended handoff pattern: a business analyst validates the prediction task in Canvas, then a data scientist takes over in Studio for production-grade optimization and deployment.

### Cost and governance

Canvas billing includes:
- **Session charges:** Per-hour cost while the Canvas app is running.
- **Training charges:** Standard Build uses Autopilot, which provisions training instances. Quick Build is significantly cheaper.
- **Storage:** Imported datasets are stored on the Domain EFS volume.

Domain administrators can control Canvas access through user profile permissions. To restrict a user from launching Canvas, remove the Canvas app permission from their profile's execution role policy.

## Connecting to Practice

Canvas demonstrates that ML model building can be accessible to non-engineers within the same governed SageMaker environment. The next topic, *Autopilot AutoML Modes*, goes deeper into the automated training system that powers Canvas's Standard Build. The final topic in this section, *Canvas vs. Autopilot*, compares the two approaches to help you decide which to recommend for different team members and use cases.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Canvas User Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas.html)** - *Docs*: Complete reference for Canvas features, supported data sources, and model management.
- **[Canvas Pricing](https://aws.amazon.com/sagemaker/canvas/pricing/)** - *Docs*: Session and training cost breakdown for Canvas workloads.

**Interactive practice**

- **[No-code ML with SageMaker Canvas Workshop](https://catalog.workshops.aws/canvas-immersion-day/en-US)** - *Interactive*: Guided lab that walks through a complete Canvas workflow from data import to batch prediction.
