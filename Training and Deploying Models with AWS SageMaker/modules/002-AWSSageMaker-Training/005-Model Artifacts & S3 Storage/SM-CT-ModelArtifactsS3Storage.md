# Model Artifacts & S3 Storage

**Estimated Time:** 10 Minutes

## Introduction

In the *Training Job Anatomy* reading you learned that when your Script Mode script finishes, SageMaker compresses everything in `/opt/ml/model/` into a file called `model.tar.gz` and uploads it to S3. That single sentence hides a lot of practical detail. Where exactly does the artifact land? What is inside it? How do you find it later? And how does this connect to what you did in the AIML Foundations module, where you saved models with `joblib.dump()` to a local file path?

This reading answers those questions. Model artifacts are the bridge between the Train stage and the Deploy stage of the SageMaker ML lifecycle. Everything downstream -- model registration, endpoint deployment, batch transform -- starts from the `model.tar.gz` sitting in S3. If you cannot find your artifact, cannot open it, or do not understand its structure, the rest of the pipeline stalls.

Think of the artifact as the finished product leaving the factory floor. The training job was the manufacturing process; S3 is the warehouse. This reading teaches you how the product gets packaged, where it is stored, and how to inspect it.

## Core Concepts

### What goes into model.tar.gz

When your training script finishes, SageMaker looks at the `/opt/ml/model/` directory inside the container and compresses its entire contents into a single `model.tar.gz` archive. Whatever files your script saved there are included -- nothing more, nothing less.

In a typical scikit-learn Script Mode script, you save one file:

```python
import joblib
import os

model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
```

The resulting `model.tar.gz` contains:

```
model.tar.gz
└── model.pkl
```

For PyTorch or TensorFlow, the structure might include multiple files (model weights, configuration files, tokenizer files):

```
model.tar.gz
├── model.pth
├── config.json
└── tokenizer.json
```

The key rule is simple: **everything you save to `SM_MODEL_DIR` ends up in the archive**. If you save extra files (metrics summaries, feature importance plots, preprocessing objects), they are included too. Be intentional about what you write there -- deployment inference code will later extract this archive and expect a specific structure.

### From local saving to SageMaker saving

In the AIML Foundations module, saving a trained model looked like this:

```python
joblib.dump(model, "models/random_forest.pkl")
```

The file landed on your laptop's filesystem. You could open it, version-control it with Git, or share it by copying the file. The limitation is that the model lives on one machine and is not automatically connected to any deployment infrastructure.

In SageMaker, the equivalent is:

```python
joblib.dump(model, os.path.join(os.environ["SM_MODEL_DIR"], "model.pkl"))
```

The file lands in the container's `/opt/ml/model/` directory. After training, SageMaker:
1. Compresses the directory into `model.tar.gz`.
2. Uploads the archive to the S3 output path you configured on the Estimator.
3. Tears down the container.

The model now lives in S3, which is durable (11 nines of durability), accessible from any AWS service, and directly consumable by SageMaker's deployment tools. The path from training to deployment is a single S3 URI.

### Where artifacts land in S3

The S3 output path follows a predictable pattern based on what you configured:

```
s3://<bucket>/<output_path_prefix>/<training-job-name>/output/model.tar.gz
```

For example, if you set `output_path="s3://my-sagemaker-bucket/training-output/"` on your Estimator, and the training job was named `rf-classifier-2026-03-22-14-30-00`, the artifact lands at:

```
s3://my-sagemaker-bucket/training-output/rf-classifier-2026-03-22-14-30-00/output/model.tar.gz
```

SageMaker appends the training job name and `/output/` automatically. This means every training job produces its artifact in its own subdirectory, so artifacts from different runs never overwrite each other.

### Finding artifacts in the console

Here is the step-by-step walkthrough for locating a model artifact through the browser.

**From the training job details page:**

1. Navigate to **SageMaker > Training > Training jobs** in the console sidebar.
2. Click the name of the completed training job.
3. Scroll to the **Output data configuration** section. You will see the **S3 output path** displayed as a clickable link.
4. Click the link. The S3 console opens, navigated directly to the output directory for that training job.
5. You will see a folder named `output/`. Click into it.
6. Inside you will find `model.tar.gz` -- your trained model artifact.

**From the S3 console directly:**

1. Search "S3" in the console search bar and open the S3 console.
2. Navigate to the bucket you configured for training output.
3. Browse into the prefix structure: `<output_path_prefix>/<training-job-name>/output/`.
4. Click `model.tar.gz` to see its details (file size, last modified date, storage class).

> **Tip:** If you have many training jobs and cannot remember the exact output path, the training job details page is always the fastest way to find the artifact. Every completed job links directly to its output.

### Downloading and inspecting artifacts

You can download the artifact from the S3 console by selecting `model.tar.gz` and clicking **Download**. To inspect its contents, extract it locally:

```bash
tar -xzf model.tar.gz
```

This produces the files your script saved to `SM_MODEL_DIR`. For a scikit-learn model, you can then load and test it locally:

```python
import joblib

model = joblib.load("model.pkl")
predictions = model.predict(X_test)
```

This inspection workflow is valuable for verifying that your training script saved the correct files before you deploy. If the archive is missing the model file or contains unexpected files, you know to fix your script's saving logic.

You can also download from the CLI:

```bash
aws s3 cp s3://my-sagemaker-bucket/training-output/rf-classifier-2026-03-22/output/model.tar.gz .
```

### Artifact versioning and organization

S3 does not automatically version your model artifacts in the way Git versions code. However, because each training job writes to its own subdirectory (keyed by the unique job name), you get implicit versioning: every training run produces a distinct artifact at a distinct path.

For more formal versioning, SageMaker provides the **Model Registry**, which you will learn about in Module 3. The Model Registry lets you register specific artifacts as versioned model packages with metadata, approval status, and deployment history. The artifact in S3 is the raw file; the Model Registry is the catalog that organizes and governs it.

For now, a practical habit is to name your training jobs descriptively (e.g., `rf-v2-feature-scaling-2026-03-22`) so you can identify artifacts by their job name when browsing S3.

### The artifact's role in deployment

Model artifacts are the input to the Deploy stage. When you create a SageMaker Model object (the precursor to an endpoint), you point it at the `model.tar.gz` in S3. SageMaker downloads the archive, extracts it into a serving container, and uses your model file to handle inference requests.

This means the structure of your artifact matters. The serving container expects specific files in specific formats. For scikit-learn, it expects a file loadable by `joblib` or `pickle`. For PyTorch, it expects model weights and an `inference.py` script. If your training script saves files in the wrong format or with unexpected names, deployment will fail.

The *Real-time Inference Endpoints* and *Invoking Endpoints* topics in Module 3 cover the deployment side in detail. For now, the key takeaway is: **what you save in `SM_MODEL_DIR` during training is exactly what the inference container will receive during deployment**. Plan your saving logic accordingly.

### SDK access to artifact paths

After a training job completes, the SDK provides direct access to the artifact location:

```python
artifact_path = estimator.model_data
print(artifact_path)
```

This returns the full S3 URI (e.g., `s3://my-sagemaker-bucket/training-output/rf-classifier-.../output/model.tar.gz`). You can pass this URI directly to `sagemaker.model.Model()` or to the Model Registry without manually constructing the path.

## Connecting to Practice

This reading completes Module 2. You now understand the full training pipeline: choosing between BYOM and Script Mode, structuring your script, configuring an Estimator, running a training job, and finding the resulting model artifact in S3. In the *Model Artifacts & S3 Storage Video*, you will see a live walkthrough of locating and inspecting an artifact. In the module lecture and assignment, you will run training jobs end to end and verify the output.

The most useful thing you can do right now is find a completed training job in your console (or run one), navigate to its output in S3, download the `model.tar.gz`, and extract it. Verify that the files inside match what your script saved. This simple check prevents many deployment failures later.

---

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Model Artifacts](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html)** - *Docs*: The official reference for how SageMaker packages and stores training output, including the container directory contract.
- **[Amazon S3 User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)** - *Docs*: Comprehensive guide to S3 concepts (buckets, prefixes, storage classes), useful for understanding where your artifacts live and how to manage them.

**Interactive practice**

- **[AWS Hands-On: Store and Retrieve a File with S3](https://aws.amazon.com/getting-started/hands-on/backup-files-to-amazon-s3/)** - *Interactive*: A free guided lab covering S3 basics (upload, download, navigate) that reinforces the artifact retrieval workflow.
- **[SageMaker Examples - Model Artifacts](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk)** - *Interactive*: Runnable notebooks demonstrating end-to-end training workflows where you can inspect the artifact output.
