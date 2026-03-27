# Week 1 Friday -- SageMaker Foundations, Training, and Deep Learning Architectures

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- SM Foundations: Ecosystem & Core Services, Studio Domains & Profiles, IAM & Least-Privilege Practices, The SageMaker ML Lifecycle, JumpStart Pre-built Models
- SM Training: BYOM & BYOS Approaches, Script Mode Structure, Estimators & Configurations, Training Job Anatomy, Model Artifacts & S3 Storage
- Deep Learning: CNNs for Image Data, Encoder-Decoder Architectures

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | SageMaker Setup + CNN Foundations | 60 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | SageMaker Training + CNN Training & Augmentation | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Encoder-Decoder Architectures + ML Lifecycle & Cleanup | 45 |
| Buffer | Open Q&A, Summary, Monday Preview | 15 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

Associates continue as ML engineers at FraudShield. Earlier this week the team trained MLPs on tabular fraud data. Now leadership wants two things: a governed SageMaker environment for cloud-based training, and a prototype image classifier for document verification (detecting forged IDs submitted during account onboarding). Friday's session covers the platform setup, the first deep learning architecture suited to images (CNNs), the mechanics of running training on SageMaker, and a second architecture pattern (encoder-decoder) that converts one sequence into another.

1. **"How do we set up our ML workspace on AWS?"** (Studio Domain, IAM roles)
2. **"What model architecture works for images?"** (CNNs)
3. **"How do we train models on SageMaker instead of our laptops?"** (Script Mode, Estimators, training jobs)
4. **"What if our task produces a sequence, not a single label?"** (Encoder-decoder)

Each stage pairs a SageMaker block with a Deep Learning block so the instructor can bounce between platform knowledge and model knowledge.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] AWS account with SageMaker access verified (instructor and all Associates)
- [ ] No pre-existing SageMaker Domain in the account (or documented plan for multi-Domain setup)
- [ ] Browser open to [console.aws.amazon.com](https://console.aws.amazon.com) and signed in
- [ ] Screen sharing enabled with font/zoom large enough for projector readability
- [ ] SageMaker pinned to the console navigation bar
- [ ] Billing & Cost Management console bookmarked for cleanup verification
- [ ] S3 bucket created with training data uploaded:
  - Bucket: `fraudshield-training-data` (or similar)
  - Prefix: `data/train/` containing `train.csv`
  - Prefix: `data/validation/` containing `validation.csv`
- [ ] Training data is a simple tabular CSV with columns including `target` (binary: 0/1 for fraud/non-fraud), 5-8 numeric feature columns
- [ ] Studio notebook open with SageMaker Python SDK installed (`pip install sagemaker`)
- [ ] Python 3.10+, numpy, and matplotlib installed in demo environment
- [ ] Companion lecture notebook (`W1-Friday-notebook.ipynb`) open and tested
- [ ] Backup diagrams ready for: CNN feature maps, encoder-decoder information flow
- [ ] This instructor guide open in a second tab or printed

### Recommended Data Preparation

Generate a synthetic fraud dataset in a notebook before the lecture (or do it live):

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "amount": np.random.exponential(500, n).round(2),
    "hour": np.random.randint(0, 24, n),
    "distance_from_home": np.random.exponential(50, n).round(2),
    "transaction_count_24h": np.random.poisson(5, n),
    "is_international": np.random.binomial(1, 0.1, n),
    "merchant_risk_score": np.random.uniform(0, 1, n).round(3),
})
data["target"] = ((data["amount"] > 800) & (data["hour"] < 6) | (data["merchant_risk_score"] > 0.85)).astype(int)

train = data.iloc[:1600]
val = data.iloc[1600:]
train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
```

Upload the CSVs to S3 before class:

```bash
aws s3 cp train.csv s3://fraudshield-training-data/data/train/train.csv
aws s3 cp validation.csv s3://fraudshield-training-data/data/validation/validation.csv
```

### Student Prerequisites

- [ ] AWS account credentials (IAM user with console access)
- [ ] Completed readings: all SM Foundations CTs, all SM Training CTs, CNNs for Image Data CT, Encoder-Decoder Architectures CT
- [ ] Browser open to AWS Console
- [ ] Studio notebook open with SageMaker SDK available
- [ ] Associates have built MLPs in Module 003

---

# STAGE 1 -- SageMaker Setup + CNN Foundations (60 min)

> **Goal:** Stand up a governed SageMaker environment and understand the first deep learning architecture designed for images.

**Exit Criteria Addressed:**
- Describe the core services in the AWS SageMaker ecosystem and their role in the ML lifecycle (Required)
- Identify the differences between SageMaker Studio, Studio Classic, and Canvas for different user personas (Required)
- Implement IAM least-privilege roles and policies for SageMaker Execution Roles (Required)
- Design a CNN architecture for image classification using convolutional, pooling, and fully connected layers (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "This week you built MLPs for tabular fraud detection. Today we level up in two directions at once: we set up SageMaker so your models run in the cloud instead of on your laptop, and we learn the first deep learning architecture that handles images -- convolutional neural networks. By the end of this lecture you will have a working SageMaker environment, you will understand how CNNs exploit spatial structure, and you will know how to launch a training job on managed infrastructure."

> "Our scenario continues: FraudShield's leadership now wants two things. First, a governed cloud ML workspace so the whole team can train and deploy models reproducibly. Second, a prototype image classifier for document verification -- detecting forged IDs in account applications. Let's start with the workspace."

---

## STEP 0 -- Set Up an AWS Budget (5 minutes)

**Pacing: live demonstration.** Walk through each click while Associates follow on their own accounts. **Every student must complete this before proceeding.**

Before touching SageMaker, set a spending guard so students are alerted before costs get out of hand.

1. In the AWS Console, search for **Billing and Cost Management** (or **Budgets**).
2. Click **Budgets** in the left sidebar, then **Create a budget**.
3. Select **Customize (advanced)** > **Cost budget** > Next.
4. Configure:

| Field | Value |
|-------|-------|
| Budget name | `SageMaker-Training-Budget` |
| Period | Monthly |
| Budget amount | **$50.00** |
| Start month | Current month |

5. Click **Next** to configure alerts.
6. Add a threshold alert:

| Field | Value |
|-------|-------|
| Threshold | 80% of budgeted amount ($40) |
| Trigger | Actual cost |
| Email recipients | Student's email address |

7. (Optional) Add a second alert at **100%** ($50) so they get a final warning.
8. Click **Next** > **Create budget**.

> **Instructor Note:** Mention that this budget covers all AWS services by default, but SageMaker endpoints and training jobs are the primary cost drivers in this curriculum. Students can scope the budget to SageMaker only by adding a filter: on the "Configure budget" page, expand **Budget scope** > **Filter** > **Service** > select **Amazon SageMaker**.

[PAUSE -- Verify every student has a budget created before moving on.]

---

## STEP 1 -- Finding SageMaker in the Console (5 minutes)

**Pacing: live demonstration.** Navigate the console while Associates follow on their own screens.

1. Sign in to the AWS Console. Point out the Console Home page layout: recently visited services, search bar, region selector.
2. Click the search bar (or press `Alt+S`). Type "SageMaker." Click **Amazon SageMaker**.
3. Walk through the left sidebar: **Studio**, **Notebook instances**, **Training**, **Inference**, **Processing**, **Pipelines**, **Governance**.

> "Each sidebar section maps to a stage of the ML lifecycle you read about. Studio is for Build. Training is for Train. Inference is for Deploy. Governance is for versioning and approval. By the end of this curriculum, you will have used every one of these sections."

4. Pin SageMaker to the navigation bar (click the star icon).

**Instructor Note:** If any student cannot find SageMaker, check their region. SageMaker availability varies by region. Recommend `us-east-1` for the broadest service coverage.

[PAUSE FOR Q&A - Ask: "Which sidebar section do you think we will use most today?"]

---

## STEP 2 -- Creating a Studio Domain (10 minutes)

**Pacing: live demonstration, step by step.** Wait for all Associates to complete each step before moving on.

1. In the SageMaker sidebar, click **Studio** (or **Admin configurations > Domains**).
2. If no Domain exists, click **Create Domain**.
3. Select **Quick setup**. Explain: "Quick setup uses your default VPC, creates a default execution role, and adds one User Profile. For production you would use Standard setup to customize the network and encryption. For learning, Quick setup is perfect."
4. Review the defaults:
   - **Domain name:** Change to `fraudshield-domain` (or leave default).
   - **Execution role:** Accept "Create a new role."
   - **VPC:** Default VPC is fine.
5. Click **Submit**. Domain creation takes 3-5 minutes.

> "While we wait: SageMaker is provisioning an EFS volume for shared storage, configuring network access within your VPC, and creating the execution role. Every User Profile in this Domain shares the same infrastructure but gets their own home directory on EFS."

**Use the wait time** to discuss Studio vs. Studio Classic vs. Canvas:

| Interface | Audience | Code? |
|-----------|----------|-------|
| Studio | ML engineers, data scientists | Yes |
| Studio Classic | Legacy users | Yes |
| Canvas | Business analysts | No |

6. When the Domain status shows **InService**, click the Domain name to view details.
7. Click **Launch > Studio** next to the profile. Let Studio load in a new tab. Point out the JupyterLab interface.

> "You are now inside SageMaker Studio. This is the IDE where you will spend most of your time in later modules. It runs entirely in the browser -- no local installation required."

[PAUSE FOR Q&A - Ask: "What three things does a Domain bundle together?" (VPC/network, default execution role, shared EFS storage)]

---

## STEP 3 -- Examining the IAM Execution Role (10 minutes)

**Pacing: live demonstration with pauses for discussion.**

1. Open a new tab. Search for "IAM" in the console search bar. Click **IAM**.
2. In the IAM sidebar, click **Roles**.
3. Search for "SageMaker." Find the auto-generated role (e.g., `AmazonSageMaker-ExecutionRole-YYYYMMDDTHHMMSS`).
4. Click the role name. Walk through:

**Trust relationships tab:**

> "This JSON says the SageMaker service is allowed to assume this role. Without this trust relationship, SageMaker cannot use the role."

Show the trust policy JSON. Point out the `sagemaker.amazonaws.com` service principal.

**Permissions tab:**

> "The attached policy `AmazonSageMakerFullAccess` grants broad permissions -- S3, ECR, CloudWatch, and more. Convenient for learning but violates least-privilege for production. If this role were compromised, the attacker could access every S3 bucket in your account."

5. Click on `AmazonSageMakerFullAccess` to show its policy document. Point out the `"Resource": "*"` patterns.

> "These wildcard resources mean this role can access ANY S3 bucket, ANY ECR repository. In production, you would scope permissions to specific resources. The assignment walks you through building a custom least-privilege role from scratch."

[PAUSE FOR Q&A - Ask: "Why is 'Resource: *' dangerous in a production environment?"]

---

## STEP 4 -- Why CNNs? From MLPs to Convolutions (8 minutes)

**Pacing: conceptual, no code yet. Transition from SageMaker setup to model architecture.**

> "Our SageMaker environment is ready. Now let's talk about the model we want to train on it. FraudShield needs an image classifier for document verification. MLPs can technically process images, but they are a poor fit."

- Recall from Module 003: MLPs flatten images into 1D vectors, losing spatial structure.
- A 32x32x3 image flattened is 3,072 inputs. An MLP with one 512-unit hidden layer needs 3,072 x 512 = 1.57M parameters in the first layer alone.
- CNNs exploit two key ideas:
  - **Local connectivity:** each neuron connects to a small spatial region (receptive field), not the entire input.
  - **Weight sharing:** the same filter (kernel) slides across the entire image, so learned features are translation-invariant.
- Three building blocks: **convolutional layers** (learn filters), **pooling layers** (downsample spatial dimensions), **fully connected layers** (final classification).

Draw or display a diagram showing a 3x3 kernel sliding across a 2D feature map, producing an output feature map. Annotate stride and padding.

**The Convolution Operation (math):**

For a single-channel input \(X\) and kernel \(K\) of size \(k \times k\), the output feature map at position \((i, j)\) is:

\[
Y(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X(i+m,\; j+n) \cdot K(m, n) + b
\]

With multiple input channels \(C_{in}\) and multiple output filters \(C_{out}\), each filter produces one output channel. Total learnable parameters for one convolutional layer: \(C_{out} \times (C_{in} \times k \times k + 1)\).

**Output size formula:** For input size \(W\), kernel size \(k\), padding \(p\), and stride \(s\):

\[
W_{out} = \frac{W - k + 2p}{s} + 1
\]

**Discussion Prompt:** "If a 3x3 kernel slides across a 32x32 image with stride 1 and no padding, what is the output size?" (30x30. With padding=1, it stays 32x32.)

---

## STEP 5 -- Exploring Image Data (7 minutes)

**Pacing: live code in the lecture notebook. Associates follow along.**

> "Before we train the CNN, let's look at the data it will process."

Run the image exploration cell from the lecture notebook. This downloads CIFAR-10, converts images to `.png` files organized by class directory, and visualizes samples. Key points to narrate:

- CIFAR-10 has 60,000 32x32 color images in 10 classes (50,000 train, 10,000 test).
- The built-in Image Classification algorithm expects images in a directory structure: one subdirectory per class, containing `.jpg`/`.jpeg`/`.png` files.
- The cell converts the raw CIFAR-10 data into this format and saves it to `cifar10_images/training/`.

---

## STEP 6 -- CNN Architecture and Transfer Learning (15 minutes)

**Pacing: conceptual walkthrough, then run the JumpStart configuration cell in the notebook.**

Instead of coding a CNN from scratch, we use SageMaker's **built-in Image Classification** algorithm. Under the hood it uses a pretrained MobileNet V2 from TensorFlow Hub and fine-tunes it on our data. This is **transfer learning**.

> "Why transfer learning? Training from scratch on a small dataset leads to overfitting. A model pretrained on ImageNet has already learned edges, textures, and shapes. We freeze those layers and retrain only the top classification layer for our 10 CIFAR-10 classes."

**MobileNet V2 architecture (conceptual):**

| Block | What It Does |
|-------|-------------|
| Input | 224x224x3 image (resized from 32x32 automatically) |
| Depthwise separable convolutions | Lightweight conv blocks that reduce parameters vs. standard Conv2d |
| Inverted residuals | Expand channels, depthwise conv, project back -- with skip connections |
| Global average pooling | Reduces spatial dims to a single vector per channel |
| **Top layer (retrained)** | Dense layer mapping to our 10 classes |

Walk through the key CNN building blocks (same ideas whether from-scratch or transfer):
- **Convolution:** learnable filters slide across the image, extracting local features. Weight sharing makes CNNs parameter-efficient.
- **ReLU:** element-wise non-linearity \(f(x) = \max(0, x)\) after each convolution.
- **Pooling:** downsamples spatial dimensions (MaxPool or Average Pool).
- **Residual / skip connections:** let gradients flow in deep networks.
- **Dropout:** zeros a fraction of activations to reduce overfitting.

Run the JumpStart configuration cell to see the model ID, training image, script URI, pretrained model URI, and default hyperparameters.

> "With the built-in algorithm, you do not write model code. SageMaker provides the pretrained model, the training script, and the container. You configure hyperparameters and point it at your data. After the break, we will launch this on SageMaker."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- SageMaker Training + CNN Training & Augmentation (55 min)

> **Goal:** Understand how to run training on SageMaker's managed infrastructure and deepen CNN knowledge with training concepts and data augmentation.

**Exit Criteria Addressed:**
- Differentiate between Bring-Your-Own-Model (BYOM) and Bring-Your-Own-Script (Script Mode) approaches (Required)
- Architect a SageMaker training script using the required Script Mode structure (main guard, parser, output paths) (Required)
- Configure a SageMaker Estimator with appropriate instance types and hyperparameters (Required)
- Describe the anatomy of a SageMaker Training Job (input channels, Docker containers, S3 locations) (Required)
- Apply data augmentation techniques to improve CNN generalization (Required)

### Instructor Opening (2 minutes)

> "We have a SageMaker environment and a CNN architecture. Now the question is: how do we actually train this on SageMaker instead of our laptop? And while we learn the training pipeline, we will also cover two more CNN concepts -- training loops and data augmentation -- that directly apply to the training job we are going to run."

---

## STEP 7 -- BYOM vs Script Mode (5 minutes)

**Pacing: whiteboard/slide discussion. No code yet.**

Draw the customization spectrum on screen:

```
Built-in Algorithms <-----> Script Mode (BYOS) <-----> Bring Your Own Container (BYOM)
   (least control)          (recommended)               (most control)
```

| Approach | You Provide | SageMaker Provides | Best For |
|----------|------------|-------------------|----------|
| Built-in algorithms | Data + hyperparameters | Everything else | Standard problems, no custom code |
| Script Mode (BYOS) | Training script (.py) | Managed container + framework | Custom training with standard frameworks |
| BYOM | Complete Docker image | Compute infrastructure only | Proprietary dependencies or frameworks |

> "Today we will use both ends of this spectrum. For our tabular fraud model, Script Mode is the right choice -- you supply your sklearn training script. For image classification, we will use the built-in algorithm and supply just data and hyperparameters."

[PAUSE FOR Q&A - Ask: "When would you choose BYOM over Script Mode?" (Custom C++ extensions, proprietary frameworks, specific CUDA versions, compliance-mandated base images)]

---

## STEP 8 -- Script Mode Structure (10 minutes)

**Pacing: line-by-line in the lecture notebook.**

> "Think back to the AIML Foundations module: you loaded a CSV, trained a model, evaluated it, and saved it with `joblib.dump()`. That script ran on your laptop. Our goal is to take that same script and run it on SageMaker's managed infrastructure."

Show the local training script (for visual comparison only, do not run):

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("data/train.csv")
X = data.drop("target", axis=1)
y = data["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print(f"Training accuracy: {accuracy_score(y, model.predict(X)):.4f}")
joblib.dump(model, "model.pkl")
```

> "Three things are hardcoded: the data path, the hyperparameters, and the model save location. SageMaker needs all three to be flexible."

Walk through the Script Mode version:

**Main guard + argparse + environment paths:**

```python
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    data = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X = data.drop("target", axis=1)
    y = data["target"]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model.fit(X, y)

    accuracy = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {accuracy:.4f}")

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
```

Draw the container directory layout:

```
/opt/ml/
|-- input/
|   +-- data/
|       |-- train/        <- SM_CHANNEL_TRAIN (your CSV lands here)
|       +-- validation/   <- SM_CHANNEL_VALIDATION
|-- model/                <- SM_MODEL_DIR (save your model here)
+-- output/
    +-- failure           <- write error message on failure
```

> "Three structural changes: (1) `if __name__` guard so SageMaker can import without executing, (2) `argparse` for hyperparameters so SageMaker can pass them on the command line, (3) environment variables for data and model paths so the same script works in any SageMaker container. The training logic itself is identical."

**Write the script to disk.** In the notebook, Associates run two cells:

1. `os.makedirs("code", exist_ok=True)` -- creates the local `code/` directory.
2. `%%writefile code/train.py` -- writes the Script Mode version of `train.py` into that directory.

SageMaker's `ModelTrainer` will upload this directory to S3 and inject it into the training container at runtime.

**Instructor Note:** Verify that Associates see the `Writing code/train.py` confirmation before proceeding to Step 9.

---

### Optional: Run Training in SageMaker Studio (before Step 9)

A companion notebook (`sagemaker_studio_train.ipynb`) is provided in the `lectures/` directory. It contains the same training logic as `train.py` broken into interactive cells that students can run inside SageMaker Studio.

**Instructor Note:** This is an optional walkthrough. Have students upload the notebook to Studio (via the JupyterLab upload button) and select the **Python 3 (Data Science 3.0)** kernel. It gives them hands-on experience with the Studio environment they set up in Step 2, and the comparison table at the end reinforces why managed training jobs are preferred for production. Remind students to **shut down the Studio kernel** when done to stop billing.

---

### Windows Line-Ending Fix (before Step 9)

**Instructor Note:** If Associates are running on Windows, they must run the SDK patch cell before launching any training jobs. The SageMaker SDK v3 generates a bash bootstrap script (`sm_train.sh`) with Windows `\r\n` line endings, which fails on the Linux training container. The patch cell in the notebook detects Windows and fixes the SDK's file-writing to use Unix `\n` endings. This only needs to be run once per environment.

---

## STEP 9 -- ModelTrainer Configuration and Training Job Launch (10 minutes)

**Pacing: live execution in the lecture notebook.**

In v3 of the SageMaker Python SDK, the framework-specific estimators (`SKLearn`, `PyTorch`, etc.) are replaced by a single unified `ModelTrainer` class.

```python
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode

session = Session()
role = get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name

training_image = retrieve("sklearn", region, version="1.2-1")

trainer = ModelTrainer(
    training_image=training_image,
    source_code=SourceCode(
        source_dir="code/",
        entry_script="train.py",
    ),
    role=role,
    hyperparameters={
        "n-estimators": "100",
        "random-state": "42",
    },
    output_data_config={
        "s3_output_path": f"s3://{bucket}/fraudshield/output/",
    },
)

trainer.train(
    input_data_config=[
        {
            "channel_name": "train",
            "data_source": f"s3://{bucket}/fraudshield/data/train/",
        }
    ],
    wait=True,
    logs="All",
)
```

Walk through the parameter-to-console mapping:

| Console Field | SDK Parameter | Our Value |
|--------------|--------------|-----------|
| Training image | `training_image` | sklearn 1.2-1 image via `image_uris.retrieve` |
| IAM role | `role` | Execution role ARN |
| Entry script | `SourceCode.entry_script` | `train.py` |
| Source dir | `SourceCode.source_dir` | `code/` |
| Instance type | Compute config | `ml.m5.large` (cheapest general-purpose) |
| Hyperparameters | `hyperparameters` | `n-estimators=100, random-state=42` |
| S3 output path | `output_data_config` | `s3://.../fraudshield/output/` |

> "The `input_data_config` list defines your data channels. The `channel_name` `train` maps to `SM_CHANNEL_TRAIN` in your script. Calling `.train()` submits the training job -- this is exactly what clicking 'Create training job' does in the console."

While the job runs (3-7 minutes), narrate the lifecycle:

> "SageMaker is provisioning an ml.m5.large instance, pulling the scikit-learn Docker container from ECR, downloading your training data from S3 into /opt/ml/input/data/train/, running your script, compressing /opt/ml/model/ into model.tar.gz, uploading it to S3, and tearing down the instance. You are billed only for this window."

---

## STEP 10 -- Preparing Data and Configuring the Built-in Algorithm (10 minutes)

**Pacing: live code in the lecture notebook.**

> "While the sklearn training job runs, let's prepare our CIFAR-10 images for the built-in Image Classification algorithm and upload them to S3."

Run the S3 upload cell to push the class-organized image directory to `s3://<bucket>/fraudshield/cifar10/training/`.

Then run the hyperparameter configuration cell. Walk through the key overrides:

1. **`epochs: 5`** -- enough to see convergence on a fine-tuning task.
2. **`train_only_top_layer: True`** -- freeze the pretrained feature extractor, only retrain the classification head. This is faster and avoids catastrophic forgetting.
3. **`augmentation: True`** -- enable data augmentation during training.
4. **`validation_split_ratio: 0.2`** -- the algorithm automatically holds out 20% for validation.

> "Notice there is no training script to write. The built-in algorithm handles the model, the training loop, and the data pipeline. We configure everything through hyperparameters."

**Discussion Prompt:** "How can you tell from the loss and accuracy curves whether overfitting is occurring?" (Training loss keeps decreasing but test accuracy plateaus or drops.)

---

## STEP 11 -- Data Augmentation (10 minutes)

**Pacing: reference the augmentation hyperparameters from Step 10.**

With the built-in algorithm, augmentation is controlled entirely through hyperparameters -- no code changes needed.

| Hyperparameter | What It Does | Our Setting |
|---------------|-------------|-------------|
| `augmentation` | Master switch for augmentation | `True` |
| `augmentation_random_flip` | Flips images horizontally, vertically, or both | `horizontal` |
| `augmentation_random_rotation` | Max rotation factor (fraction of 2*pi) | `0.2` |
| `augmentation_random_zoom` | Max zoom factor | `0.1` (default) |

The built-in algorithm applies augmentation **only** to training data during fine-tuning. The validation split is never augmented.

Compare this to a from-scratch approach where you write transform pipelines manually. The built-in algorithm abstracts this into simple hyperparameter toggles.

**Q&A Checkpoint (2 min):** "When would horizontal flip be a bad augmentation choice?" (Digit recognition -- a flipped 6 becomes a 9.)

> "Residual connections are covered in your reading (CNNs CT, Step 1.6). They use skip connections to enable gradient flow in very deep networks."

### Optional: Run CNN Training in SageMaker Studio (before Step 11b)

A companion notebook (`sagemaker_studio_cnn.ipynb`) is provided in the `lectures/` directory. Students can upload it to SageMaker Studio to run the full CNN image classification workflow interactively: data preparation, S3 upload, hyperparameter configuration, and training job submission. Same kernel recommendation (Python 3, Data Science 3.0). Remind students to shut down the kernel when done.

---

## STEP 11b -- Launch Built-in Image Classification Training (5 minutes)

**Pacing: live execution in the lecture notebook.**

> "Now we launch the built-in Image Classification job. SageMaker retrieves the pretrained MobileNet V2 model, loads our CIFAR-10 images, fine-tunes the top layer, and saves the result."

```python
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import InputData

js_config = JumpStartConfig(model_id=MODEL_ID, model_version=MODEL_VERSION)

ic_trainer = ModelTrainer.from_jumpstart_config(
    jumpstart_config=js_config,
    hyperparameters=custom_hp,
    role=role,
    sagemaker_session=session,
    output_data_config={"s3_output_path": f"s3://{bucket}/fraudshield/cnn-output/"},
)

ic_trainer.train(
    input_data_config=[
        InputData(channel_name="training", data_source=training_s3_uri),
    ],
    wait=True,
    logs="All",
)
```

Key differences from the sklearn Script Mode job:
- Uses `ModelTrainer.from_jumpstart_config` instead of manual `ModelTrainer` construction.
- SageMaker provides the training image, the training script (`transfer_learning.py`), and the pretrained model weights.
- `input_data_config` points to a directory of class-organized images rather than a CSV.
- No custom training script was written.

---

## STEP 12 -- Training Job Anatomy and Artifacts (5 minutes)

**Pacing: interactive consolidation. Reference both jobs.**

> "Let's compare our two training jobs and map what happened to the seven-step lifecycle."

Both jobs follow the same lifecycle:

| Step | What Happened | sklearn Job (Script Mode) | Image Classification (Built-in) |
|------|--------------|---------------------------|-------------------------------|
| 1. Provisioning | Allocated instance | `ml.m5.large` | `ml.p3.2xlarge` (GPU) |
| 2. Container pull | Pulled framework image from ECR | scikit-learn 1.2-1 | TensorFlow 2.9 |
| 3. Data download | Data into `/opt/ml/input/data/` | `train.csv` from S3 | CIFAR-10 images from S3 |
| 4. Script execution | Ran entry script | Your `train.py` | SageMaker's `transfer_learning.py` |
| 5. Artifact upload | `/opt/ml/model/` -> `model.tar.gz` -> S3 | `model.pkl` | TF SavedModel |
| 6. Logging | stdout/stderr -> CloudWatch | accuracy output | epoch metrics |
| 7. Teardown | Terminated instance | Status: Completed | Status: Completed |

Show the artifact paths from the SDK:

```python
print("sklearn model:", trainer.model_data)
print("Image Classification model:", ic_trainer.model_data)
```

> "Every training job stores its artifact under `<output_path>/<job-name>/output/model.tar.gz`. This S3 URI is the bridge between Train and Deploy -- you will pass it to Model Registry in the Deployment module."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Encoder-Decoder Architectures + ML Lifecycle & Cleanup (45 min)

> **Goal:** Understand the seq2seq encoder-decoder pattern and consolidate the SageMaker ML lifecycle with JumpStart and cleanup discipline.

**Exit Criteria Addressed:**
- Architect a seq2seq encoder-decoder model for sequence translation (Required)
- Describe the five stages of the SageMaker ML Lifecycle (Prep, Build, Train, Deploy, Monitor) (Required)
- Deploy a pre-built model from SageMaker JumpStart using the AWS Management Console (Required)
- Implement resource cleanup steps (deleting endpoints and models) to avoid unnecessary billing (Required)

### Instructor Opening (2 minutes)

> "In Stages 1 and 2 we covered CNNs: one input image produces one label. But many tasks require a sequence of outputs -- translation, summarization, chatbots. The encoder-decoder pattern solves this. We will cover it now, and then close with the SageMaker lifecycle and cleanup discipline."

---

## STEP 13 -- LSTM Primer for Encoder-Decoder (5 minutes)

**Pacing: conceptual bridge. Just enough LSTM background for encoder-decoder to make sense. The full RNN/LSTM/GRU treatment is Monday.**

> "The encoder-decoder architecture uses LSTM cells internally. Monday's reading and notebook cover RNNs, LSTMs, and GRUs in full detail. For now, here is what you need to know."

**What an LSTM does:**
- An LSTM processes a sequence one step at a time, like reading a sentence word by word.
- At each step, it takes an input token and its previous internal state, and produces an updated state.
- It maintains two state vectors: the **hidden state** \(h_t\) (a working summary) and the **cell state** \(c_t\) (a long-term memory highway).
- After processing the entire sequence, the final states \((h_T, c_T)\) summarize the input.

**Why LSTM instead of a simpler recurrence:**
- A basic recurrent cell multiplies gradients by the same weight matrix at every time step, causing gradients to vanish on long sequences.
- LSTMs use **gates** (learned, element-wise controls) that let gradients flow through the cell state via addition rather than repeated multiplication.
- The result: LSTMs can capture dependencies across dozens or hundreds of time steps.

> "You do not need to know the gate equations today. What matters is: LSTM reads a sequence, produces a final hidden state and cell state that summarize it. The encoder uses this to compress input; the decoder uses the encoder's final states to generate output. Monday goes deep on the gate mechanics."

---

## STEP 14 -- The Encoder-Decoder Idea (8 minutes)

**Pacing: conceptual. Draw the architecture before pseudocode.**

- So far, our neural networks processed an input and produced a single output (classification).
- Many tasks require **sequence-to-sequence** output: machine translation (English to French), summarization (long text to short text), chatbots (question to answer).
- The **encoder-decoder** architecture solves this:
  1. **Encoder:** reads the entire input sequence, compresses it into a fixed-size context vector (the final hidden state).
  2. **Decoder:** takes the context vector and generates the output sequence one token at a time.

Draw the architecture:

```
Encoder:  x_1, x_2, ..., x_n --> [LSTM] --> context vector (h_n, c_n)
                                                  |
Decoder:  <SOS> --> [LSTM] --> y_1 --> [LSTM] --> y_2 --> ... --> <EOS>
```

- The context vector is a **bottleneck**: it must capture everything the decoder needs to know about the input. This limitation motivates attention mechanisms (covered in Module 005).
- For today, we use a simple task: **reversing digit strings** (e.g., "1 2 3 4" -> "4 3 2 1"). This isolates the architecture from language complexity.

---

## STEP 15 -- Reversal Data + Encoder + Decoder Design (12 minutes)

**Pacing: live code for data generation, then pseudocode walkthrough for encoder/decoder.**

Run the reversal data generation cell from the lecture notebook. Explain:
- **Encoder input:** the original digit string, padded.
- **Decoder input:** SOS token + reversed digits, padded (teacher forcing -- decoder sees the correct previous token during training).
- **Decoder target:** reversed digits + EOS token, padded (what the decoder should predict).

**Encoder pseudocode:**

```
MODEL Encoder(vocab_size, embed_dim, hidden_dim):
  LAYERS:
    Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
    LSTM(input_size=embed_dim, hidden_size=hidden_dim)

  FORWARD PASS:
    embedded              <- Embedding(input_sequence)
    all_outputs, (h_n, c_n) <- LSTM(embedded)
    RETURN (h_n, c_n)     # context vector only
```

- The encoder reads the input and produces only the final hidden state and cell state.
- `padding_idx=PAD_TOKEN` ensures padding tokens get zero embeddings.

**Decoder pseudocode:**

```
MODEL Decoder(vocab_size, embed_dim, hidden_dim):
  LAYERS:
    Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
    LSTM(input_size=embed_dim, hidden_size=hidden_dim)
    FullyConnected(hidden_dim, vocab_size)

  FORWARD PASS:
    embedded                    <- Embedding(decoder_input)
    output, (h_n, c_n)         <- LSTM(embedded, initial_hidden)  # initial_hidden = context vector
    logits                     <- FC(output)
    RETURN logits, (h_n, c_n)
```

- The decoder receives the context vector as its initial hidden state.
- During training, we use **teacher forcing:** feed the correct previous token at each step.
- At inference, we feed the model's own predictions back in (**autoregressive decoding**).

---

## STEP 16 -- Seq2Seq Assembly and Training Concepts (5 minutes)

**Pacing: pseudocode walkthrough.**

```
MODEL Seq2Seq(encoder, decoder):
  FORWARD PASS:
    context_vector  <- encoder.forward(source_sequence)
    logits, _       <- decoder.forward(target_input, context_vector)
    RETURN logits

INSTANTIATION:
    embed_dim  = 16
    hidden_dim = 64
    encoder  <- Encoder(NUM_TOKENS, embed_dim, hidden_dim)
    decoder  <- Decoder(NUM_TOKENS, embed_dim, hidden_dim)
    seq2seq  <- Seq2Seq(encoder, decoder)
```

> "The Seq2Seq model is two modules composed together. The encoder compresses; the decoder generates. This pattern is the foundation of modern NLP -- GPT is a decoder, BERT is an encoder, and T5 is an encoder-decoder."

Training uses `CrossEntropyLoss(ignore_index=PAD_TOKEN)` so padding positions do not contribute to the loss. Token-level accuracy is computed on non-padded positions only.

**Q&A Checkpoint (2 min):** "What is the main limitation of compressing the entire input into a single context vector?" (Information bottleneck. Long or complex inputs lose detail. This motivates attention mechanisms in Module 005.)

---

## STEP 17 -- JumpStart Quick Demo (5 minutes)

**Pacing: live demonstration. Transition back to SageMaker.**

> "We have covered two architectures and the SageMaker training pipeline. Let's now see the other way to get a model running -- JumpStart, which gives you pre-built models with one click."

1. Switch to the Studio browser tab.
2. Navigate to **JumpStart** in the left sidebar.
3. Browse the model catalog. Point out categories: Text Generation, Image Classification, Tabular, Sentence Embeddings.
4. Select a lightweight model that deploys on `ml.m5.large`.
5. Set endpoint name to `fraudshield-demo-endpoint`. Click **Deploy**.

> "JumpStart bundles the Model, Endpoint Configuration, and Endpoint creation into one action. In the Deployment module, you will do these three steps manually. For now, just see that SageMaker can deploy a model with minimal effort."

6. While deployment runs (or after completion), test the endpoint with the sample notebook or explain the invocation pattern.

---

## STEP 18 -- ML Lifecycle Mapping (5 minutes)

**Pacing: interactive whiteboard/discussion exercise.**

> "Let's consolidate everything we did today against the five lifecycle stages."

| Lifecycle Stage | What We Did Today | Console Location |
|----------------|-------------------|-----------------|
| **Prepare** | Generated synthetic fraud data, uploaded to S3 | S3, Processing |
| **Build** | Created Studio Domain, explored CNN architecture via JumpStart | Studio, JumpStart |
| **Train & Tune** | Launched sklearn + built-in Image Classification jobs, viewed artifacts | Training jobs |
| **Deploy** | Deployed JumpStart model to endpoint | Inference > Endpoints |
| **Monitor** | (Coming in later modules) | Model monitoring |
| **Govern** | Created IAM execution role | IAM console, Domain settings |

> "We touched every stage except Monitor. Modules 2-4 fill in the details. By the end of the curriculum, you will have hands-on experience with every cell in this table."

---

## STEP 19 -- Mandatory Cleanup (5 minutes)

**Pacing: live demonstration. EVERY student must complete this step.**

> "This is the most important step of the entire lecture. If you skip this, you will be charged for every minute the endpoint runs."

**Console cleanup:**

1. Navigate to **Inference > Endpoints**. Select `fraudshield-demo-endpoint`. Click **Actions > Delete**. Confirm.
2. Navigate to **Inference > Models**. Select the associated model. Click **Actions > Delete**. Confirm.
3. Navigate to **Inference > Endpoint configurations**. Select the associated config. Click **Actions > Delete**. Confirm.
4. Navigate to **S3 console**. Open the default SageMaker bucket. Delete the `fraudshield/` prefix (training data, model artifacts, CIFAR-10 images).

> "Delete in this order: endpoint first (stops billing immediately), then model, then configuration. Always verify the endpoint is gone from the list."

5. Check **Billing & Cost Management**. Show the current charges.

> "Make this a habit: every time you deploy anything, check billing after cleanup."

**Teaching Note:** Walk around the room (or monitor screen shares) to verify every student has deleted their endpoint. This is non-negotiable.

---

## Wrap-up & Q&A Buffer (15 minutes)

### Summary (5 minutes)

> "Today you accomplished three things. First, you set up a governed SageMaker environment: Studio Domain, IAM execution role, and you understand least-privilege at a conceptual level (the assignment goes hands-on with custom role creation). Second, you learned CNNs -- convolutional layers exploit spatial structure with local filters and weight sharing, data augmentation improves generalization, and the training loop follows the same pattern as Module 003. Third, you ran a training job on SageMaker using Script Mode and saw where the model artifact lands in S3. And you saw the encoder-decoder pattern for sequence-to-sequence tasks."

### Monday Preview (2 minutes)

> "Monday covers RNNs, LSTMs, and GRUs in full detail -- the architectures that power the encoder and decoder we just saw. You will also begin SageMaker Deployment (Model Registry, endpoints) and MLOps (pipelines, CI/CD). Read the RNNs and LSTMs & GRUs concept threads before Monday."

### Open Q&A (8 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Student cannot find SageMaker | Check the region selector (top-right). Switch to `us-east-1`. |
| Domain creation fails | Usually a VPC/subnet issue. Verify the default VPC exists. |
| JumpStart model unavailable | Model catalog changes. Have 2-3 backup models that deploy on `ml.m5.large`. |
| Studio takes too long to load | First load can take 3-5 minutes. This is normal. |
| Student forgets to delete endpoint | Walk over and help immediately. Check billing together. |
| IAM role creation permission denied | Student's IAM user may lack `iam:CreateRole` permission. |
| `ModuleNotFoundError: sagemaker` | Install with `pip install sagemaker` in the notebook. |
| Training job stuck on "Starting" for 10+ minutes | Normal for first job in a new account/region. Container is being pulled for the first time. |
| `FileNotFoundError` during training | Verify the S3 path in `.fit()` matches where data was uploaded. Check channel names. |
| `ClientError: Access Denied` | The execution role lacks S3 permissions. Attach the appropriate policy in IAM. |
