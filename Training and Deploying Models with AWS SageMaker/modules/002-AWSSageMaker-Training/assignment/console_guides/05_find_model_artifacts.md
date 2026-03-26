# Guide 5: Find and Verify Model Artifacts

After a training job completes, SageMaker compresses everything your script saved to `/opt/ml/model/` into a `model.tar.gz` archive and uploads it to S3. Knowing exactly where this artifact lives and what it contains is critical -- this artifact is what you will deploy as an endpoint in Module 3.

---

## Steps

### Step 1 -- Identify the Artifact Path from the Training Job

1. Go to **SageMaker** -> **Training** -> **Training jobs**.
2. Click on your completed training job.
3. Scroll to the **Output data configuration** section.
4. Note the **S3 output path**. Combine it with the job name to get the full artifact path:

```
s3://<bucket>/output/<training-job-name>/output/model.tar.gz
```

For example:
```
s3://fraudshield-training-data-cj/output/fraudshield-rf-2026-03-22-14-30-00-123/output/model.tar.gz
```

### Step 2 -- Navigate to the Artifact in S3

1. Open the **S3 console** (search for "S3" in the top search bar).
2. Click on your `fraudshield-training-data-<initials>` bucket.
3. Navigate the prefix path:
   - Click `output/`
   - Click `<your-training-job-name>/`
   - Click `output/`
4. You should see `model.tar.gz` listed.

### Step 3 -- Examine the Artifact Details

1. Click on `model.tar.gz`.
2. Note the following on the object details page:
   - **Size:** How large is the model artifact?
   - **Last modified:** When was it uploaded?
   - **S3 URI:** Copy this -- you will need it for Model Registry in Module 3
   - **Storage class:** Standard (default)

### Step 4 -- Understand the Artifact Structure

The `model.tar.gz` file contains everything your training script saved to the `/opt/ml/model/` directory. For our script, that is:

```
model.tar.gz
└── model.pkl        (serialized RandomForestClassifier via joblib)
```

In more complex scenarios, the archive might contain:
- Multiple model files (e.g., `model.pkl` + `scaler.pkl`)
- Configuration files (e.g., `feature_names.json`)
- Preprocessing artifacts (e.g., `label_encoder.pkl`)

SageMaker does not care about the internal structure -- it simply compresses and uploads whatever is in `/opt/ml/model/`. It is the **inference script** (in Module 3) that knows how to load and use these files.

### Step 5 -- Understand Implicit Versioning

SageMaker's artifact path pattern provides implicit versioning:

```
s3://<bucket>/output/<job-name-1>/output/model.tar.gz  ← version 1
s3://<bucket>/output/<job-name-2>/output/model.tar.gz  ← version 2
s3://<bucket>/output/<job-name-3>/output/model.tar.gz  ← version 3
```

Each training job gets its own prefix (the job name includes a timestamp), so artifacts from different runs never overwrite each other. In Module 3, you will use the **Model Registry** for formal versioning with approval workflows on top of this implicit file-level versioning.

### Step 6 -- (Optional) Download and Inspect Locally

If you want to see inside the archive:

1. Click **Download** on the S3 object details page to download `model.tar.gz`.
2. On your local machine, extract it:
   ```
   tar -xzf model.tar.gz
   ```
3. You should see `model.pkl` -- the serialized scikit-learn model.
4. You could load it in Python: `model = joblib.load("model.pkl")` and call `model.predict(...)`.

---

## Presentation Checkpoint

Be prepared to show:
- The `model.tar.gz` file in S3, navigated to via the prefix path
- The full S3 URI of the artifact
- Explain: What is the S3 path pattern for model artifacts? (`<output_path>/<job_name>/output/model.tar.gz`)
- Explain: What does `model.tar.gz` contain? (Everything saved to `/opt/ml/model/` during training -- in our case, `model.pkl`)
- Explain: How does SageMaker handle versioning of artifacts? (Each training job gets a unique name with a timestamp, creating a unique S3 prefix. The Model Registry in Module 3 adds formal version tracking on top of this.)

---

## Key Concepts

- **Model Artifact (`model.tar.gz`):** A compressed archive of everything in `/opt/ml/model/`. This is the portable, deployable unit that connects training to inference.
- **Implicit Versioning:** Each training job creates a unique S3 prefix (via the job name), so artifacts are never overwritten.
- **Artifact Path Pattern:** `<output_path>/<training-job-name>/output/model.tar.gz` -- this pattern is consistent across all SageMaker training jobs.

---

## AIML Connection

The *ML Lifecycle & Reproducibility* reading emphasized that models must be reproducible and traceable. The S3 artifact path ties directly back to a specific training job (via the job name), which in turn records the exact hyperparameters, data paths, and code version used. This chain of evidence from data to artifact is the foundation of reproducible ML.
