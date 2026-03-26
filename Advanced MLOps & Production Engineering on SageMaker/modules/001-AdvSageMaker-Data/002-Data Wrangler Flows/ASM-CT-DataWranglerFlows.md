# Data Wrangler Flows

**Estimated Time:** 10 Minutes

## Introduction

In the foundational SageMaker skill, data preparation happened locally: you loaded a CSV into a pandas DataFrame, encoded categorical columns, and split into train/test sets inside a notebook or training script. That approach works when you know exactly which transformations to apply, but in practice, data preparation is an iterative discovery process. You explore column distributions, identify missing values, try different encoding strategies, and often need to join data from multiple sources before you even begin training.

SageMaker Data Wrangler is a visual data preparation tool built into Studio that accelerates this exploration phase. Instead of writing boilerplate pandas code to profile your data, you build a **Flow** -- a directed graph of transformation steps that you configure through a point-and-click interface. Data Wrangler generates the underlying PySpark or pandas code for you, and you can export the entire Flow as a processing job, a Pipeline step, or a standalone script when you are ready to operationalize it.

This reading covers how Data Wrangler Flows work architecturally, how to create and configure them in the Studio console, and where they fit in the broader MLOps lifecycle.

## Core Concepts

### What is a Data Wrangler Flow?

A Flow is a `.flow` file stored in your Studio EFS home directory. It defines a sequence of data transformations as a directed acyclic graph (DAG). Each node in the graph represents either a **data source** (S3, Athena, Redshift, Snowflake, or a live database connection) or a **transform step** (rename columns, encode categoricals, handle missing values, apply custom pandas/PySpark code, etc.).

When you open a Flow in Studio, Data Wrangler renders it visually as a node-and-edge diagram. You add steps by clicking the "+" icon after an existing node and selecting from a library of built-in transforms or writing custom code.

### Creating a Flow in Studio

1. Open **SageMaker Studio** and click **Data Wrangler** from the launcher (or navigate to **File > New > Data Wrangler Flow**).
2. Data Wrangler launches a compute instance to power the interactive session. By default this uses an `ml.m5.4xlarge`. For cost control, you can change the instance type in the Flow's compute settings.
3. **Import data:** Click the **Import** tab. Select a data source (S3 is the most common). Browse to your bucket, select the file, and configure the import settings (file type, delimiter, sampling). Data Wrangler loads a sample (default: first 50,000 rows) for interactive exploration.
4. **Add transforms:** After importing, click the "+" after the import node and choose **Add transform**. The transform panel shows categories: Format, Handle Missing, Encode Categorical, Parse Column, Custom Transform, and more. Each transform previews its effect on the sample data before you apply it.
5. **Chain transforms:** Each applied transform adds a new node to the Flow. You can branch the graph (apply different transforms to different columns in parallel) or keep it linear.

### Built-in transform library

Data Wrangler includes over 300 built-in transforms. The most commonly used categories:

| Category | Examples | When to use |
| :--- | :--- | :--- |
| Handle Missing | Impute with mean/median/mode, drop rows, fill with custom value | Cleaning raw data before training |
| Encode Categorical | One-hot, ordinal, target encoding | Preparing string columns for ML algorithms |
| Feature | Standard scaler, min-max scaler, log transform | Normalizing numeric features |
| Parse Column | Extract date components, split strings, flatten JSON | Structuring semi-structured data |
| Custom Transform | Pandas, PySpark, or SQL code blocks | Anything not covered by built-in transforms |

Custom transforms are particularly powerful: you write a Python function that receives a DataFrame and returns the transformed DataFrame. This lets you embed domain-specific logic while keeping it inside the visual Flow for traceability.

### Data profiling and insights

Before adding transforms, you should profile your data. Click the **Data Quality and Insights Report** option in the Flow to generate an automated report that includes:

- Column-level statistics (mean, median, standard deviation, unique values, missing percentage).
- Distribution histograms for numeric columns.
- Target leakage warnings if you specify a target column.
- Duplicate row detection.
- Anomaly and outlier detection.

This report serves the same purpose as manual EDA in a notebook but is generated automatically and can be regenerated after each transformation step to verify the effect.

### Flow architecture and compute

Data Wrangler runs on a dedicated ML instance within your Studio Space. The interactive session (the "Data Wrangler app") is separate from your JupyterLab notebook compute. This matters for two reasons:

1. **Cost:** The Data Wrangler instance runs as long as the Flow tab is open. Always close or shut down the Data Wrangler app when you are not actively using it.
2. **Scaling:** The interactive session processes a sample of your data. When you export the Flow to a Processing Job (covered in the next topic, *Feature Store Architecture*, and in *Exporting Flows to Pipelines*), SageMaker runs the full dataset on a cluster of instances that you configure at export time.

### Saving and versioning Flows

Flow files (`.flow`) are saved to your Studio EFS home directory by default. For version control, download the `.flow` file and commit it to your Git repository. The file is JSON under the hood and diffs reasonably well. You can also share Flows with teammates by copying the `.flow` file to a shared S3 location and importing it into their Studio session.

## Connecting to Practice

Data Wrangler is the starting point for the data pipeline you will build across this module. The next topic, *Feature Store Architecture*, shows where the cleaned and transformed features go for storage and serving. *Exporting Flows to Pipelines* covers how to promote a manual Flow into an automated Pipeline step. In the module lecture and assignment, you will build a Flow that ingests raw data from S3, applies transforms, profiles the result, and exports the Flow for pipeline integration.

## Further Learning & Resources

**Documentation and reading**

- **[Data Wrangler Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html)** - *Docs*: Complete reference for Data Wrangler features, data source connectors, and transform APIs.
- **[Data Wrangler Pricing](https://aws.amazon.com/sagemaker/data-wrangler/pricing/)** - *Docs*: Instance pricing for interactive sessions and exported processing jobs.

**Interactive practice**

- **[Prepare ML Data with Amazon SageMaker Data Wrangler](https://catalog.workshops.aws/sagemaker-data-wrangler/en-US)** - *Interactive*: Hands-on workshop that walks through a complete Data Wrangler flow from import to export.
