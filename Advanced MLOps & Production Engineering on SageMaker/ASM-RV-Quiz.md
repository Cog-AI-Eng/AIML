# Advanced MLOps & Production Engineering on SageMaker Review Quiz

**Activity Name:** ASM-RV-Quiz
**Display Name:** Advanced MLOps & SageMaker Production Engineering Review Quiz
**Duration:** 45 min
**Total Questions:** 36
**Question Types:** Multiple Choice (26), True/False (6), Matching (4)
**Difficulty Distribution:** Intermediate (12), Advanced (18), Expert (6)

---

## Questions

---

### Question 1 -- MCQ | Module 1 | Intermediate | Conceptual

**What distinguishes the Standard Setup option from Quick Setup when creating a SageMaker Studio Domain?**

A) Standard Setup provisions GPU instances while Quick Setup only provisions CPU instances
B) Standard Setup allows you to specify a custom VPC, subnets, security groups, and encryption settings, while Quick Setup uses the default VPC and auto-creates a default execution role
C) Standard Setup is only available in us-east-1 while Quick Setup works in all regions
D) Standard Setup creates a Studio Classic domain while Quick Setup creates a Studio domain

**Correct Answer: B**

**Rationale:** Standard Setup gives administrators control over networking (VPC, subnets, security groups), encryption (KMS keys), and IAM (custom execution roles), which are critical for production environments. Quick Setup abstracts these decisions by using the default VPC and auto-creating a permissive execution role. GPU provisioning (A) is determined by instance type selection, not setup mode. Both modes are available in all SageMaker-supported regions (C). Both create the same Studio domain type (D).

---

### Question 2 -- MCQ | Module 1 | Advanced | Scenario-Based

**A data engineer creates a Data Wrangler flow that applies one-hot encoding, standard scaling, and missing value imputation to a 500,000-row dataset. She wants to run this flow on the full dataset as part of a nightly retraining pipeline. What is the most operationally efficient approach?**

A) Manually open Data Wrangler every night and click "Run All"
B) Export the Data Wrangler flow to a SageMaker Pipeline Processing Step, then schedule the pipeline on a cron trigger
C) Download the data to a local machine, run the transformations in pandas, and re-upload to S3
D) Create a Lambda function that opens Data Wrangler and clicks the Run button programmatically

**Correct Answer: B**

**Rationale:** Data Wrangler flows can be exported as SageMaker Pipeline Processing Steps, converting interactive transforms into repeatable, schedulable pipeline stages. This approach preserves the exact transform logic while enabling automation. Manual execution (A) does not scale and introduces human error. Local processing (C) breaks the cloud-native workflow and cannot handle large datasets efficiently. Lambda cannot interact with Data Wrangler's UI (D).

---

### Question 3 -- MCQ | Module 1 | Advanced | Scenario-Based

**A team creates a Feature Store Feature Group with both an Online Store (DynamoDB) and an Offline Store (S3/Glue). A real-time fraud detection endpoint needs sub-10ms feature lookups at inference time, while the training pipeline needs full historical feature data. Which store serves each use case?**

A) Online Store for both -- DynamoDB handles historical queries efficiently
B) Offline Store for both -- S3 is cheaper and faster
C) Online Store for real-time inference lookups; Offline Store for historical training data retrieval
D) Online Store for training data; Offline Store for inference lookups

**Correct Answer: C**

**Rationale:** The Online Store is backed by DynamoDB and provides single-digit millisecond lookups by record key, making it ideal for real-time inference feature retrieval. The Offline Store persists data to S3 in Parquet format with an AWS Glue Data Catalog, enabling efficient full-table scans and time-travel queries needed for training dataset assembly. DynamoDB is not designed for full historical scans (A). S3 cannot provide sub-10ms point lookups (B). Option D reverses the correct mapping.

---

### Question 4 -- T/F | Module 1 | Intermediate

**True or False: SageMaker Canvas allows business analysts with no coding experience to build ML models using a visual, drag-and-drop interface, while Autopilot provides a programmatic AutoML experience that automatically selects algorithms, tunes hyperparameters, and generates candidate notebooks.**

**Correct Answer: True**

**Rationale:** Canvas is a no-code, visual interface designed for business users who can import data, select a target column, and build models without writing code. Autopilot is an AutoML service that programmatically evaluates multiple algorithms, performs hyperparameter tuning, and generates Jupyter notebooks showing its methodology. Both automate model building but target different user personas and access patterns.

---

### Question 5 -- MCQ | Module 1 | Expert | Scenario-Based

**A Feature Group is configured with Online Store enabled. After ingesting 10,000 records, the team discovers that the Online Store only contains the latest value for each record key, not the full history. They need the full history for a regulatory audit. Where should they look?**

A) The Online Store automatically stores all versions; they need to change the query parameters
B) The Offline Store, which appends records with timestamps and preserves all historical versions
C) CloudWatch Logs, which stores all Feature Store write operations
D) The SageMaker Model Registry, which tracks feature lineage

**Correct Answer: B**

**Rationale:** The Online Store maintains only the latest value per record key (it is a key-value store optimized for real-time lookups). The Offline Store appends every ingested record with an event time and write time, preserving full history in S3 Parquet files. This append-only design makes the Offline Store the correct source for historical audits. CloudWatch Logs (C) record API calls but not feature values. The Model Registry (D) tracks model artifacts, not feature data.

---

### Question 6 -- Matching | Module 1 | Advanced

**Match each SageMaker data preparation tool to its primary use case.**

| # | Tool | | Use Case |
|---|------|---|----------|
| 1 | Data Wrangler | | A) Centralized feature storage with online/offline access patterns |
| 2 | Feature Store | | B) No-code ML model building for business analysts |
| 3 | Canvas | | C) Visual data exploration, profiling, and transformation pipeline design |
| 4 | Autopilot | | D) Programmatic AutoML with algorithm selection and candidate notebook generation |

**Correct Answers: 1-C, 2-A, 3-B, 4-D**

**Rationale:** Data Wrangler provides a visual interface for data profiling and transformation (C). Feature Store is the centralized repository for ML features with dual online/offline access (A). Canvas targets non-technical users with a no-code experience (B). Autopilot automates the full model building process programmatically and generates explainable notebooks (D).

---

### Question 7 -- MCQ | Module 2 | Intermediate | Conceptual

**When using SageMaker's built-in XGBoost algorithm in Algorithm Mode (not Script Mode), how are hyperparameters specified?**

A) In a Python training script passed as the entry_point
B) As key-value pairs in the training job configuration through the console or Estimator API, without writing any training code
C) In a YAML configuration file uploaded to S3
D) Through environment variables set on the training instance

**Correct Answer: B**

**Rationale:** Algorithm Mode uses SageMaker's pre-packaged algorithm containers that require no user-provided training script. Hyperparameters are passed directly as key-value pairs through the console's training job form or the Estimator's `hyperparameters` dictionary. This contrasts with Script Mode where you write and supply a training script (A). SageMaker does not use YAML config files (C). While environment variables exist inside the container (D), hyperparameters are not configured through them by the user.

---

### Question 8 -- MCQ | Module 2 | Advanced | Scenario-Based

**A team trains an RCF (Random Cut Forest) model on 6 months of transaction amounts to detect anomalies. The model flags transactions above $5,000 as anomalous, but the business considers transactions above $2,000 during off-hours (midnight to 5 AM) as suspicious too. What should the team adjust?**

A) Increase the number of trees in the forest to capture time-of-day patterns
B) Add time-of-day as a second feature dimension so the model can learn the joint distribution of amount and time, then re-evaluate anomaly score thresholds
C) Switch from RCF to a supervised classification algorithm since the business has labeled examples
D) Lower the RCF anomaly score threshold to 0, which will flag all transactions

**Correct Answer: B**

**Rationale:** RCF is an unsupervised algorithm that detects anomalies based on the distribution of input features. With only the amount feature, it cannot distinguish time-of-day patterns. Adding time-of-day as a feature allows the model to learn that high-amount + off-hours is rare (anomalous) even when high amounts alone are more common. Simply adding trees (A) increases model capacity but does not address the missing feature. Switching to supervised learning (C) is premature when the unsupervised approach just needs richer features. A zero threshold (D) would flag everything, defeating the purpose.

---

### Question 9 -- MCQ | Module 2 | Advanced | Scenario-Based

**An ML engineer configures an HPO job with Bayesian strategy, 20 max trials, and 4 max parallel jobs. After 20 trials, the best objective metric is 0.87 F1. She reconfigures with Random strategy, same budget. Under what condition would Random search likely outperform Bayesian?**

A) When the hyperparameter search space is very small (only 2-3 parameters with narrow ranges)
B) When the objective function landscape is highly irregular with many local optima and no smooth gradient for the surrogate model to exploit
C) Random search always outperforms Bayesian search because it explores more of the space
D) When using SageMaker's built-in algorithms instead of Script Mode

**Correct Answer: B**

**Rationale:** Bayesian optimization builds a probabilistic surrogate model of the objective function and exploits smooth gradients to guide the search. When the landscape is highly irregular with many discontinuities and local optima, the surrogate model's assumptions break down, and its guided exploration can get trapped. Random search explores the space uniformly and may stumble upon better regions by chance in these scenarios. Small search spaces (A) favor Bayesian since fewer trials are needed. Random does not always outperform Bayesian (C). The strategy choice is independent of Algorithm Mode vs. Script Mode (D).

---

### Question 10 -- MCQ | Module 2 | Expert | Scenario-Based

**An HPO job is configured with `max_parallel_jobs=10` and Bayesian strategy. Compared to `max_parallel_jobs=1`, what trade-off does high parallelism introduce?**

A) No trade-off; more parallelism always produces better results faster
B) Higher parallelism reduces wall-clock time but degrades Bayesian optimization effectiveness because the surrogate model has fewer completed trials to learn from when selecting the next batch of configurations
C) Higher parallelism increases S3 storage costs but does not affect optimization quality
D) Higher parallelism requires GPU instances, increasing cost

**Correct Answer: B**

**Rationale:** Bayesian optimization is inherently sequential -- each trial's result informs the next trial's configuration. With high parallelism, the surrogate model must propose 10 configurations simultaneously before any complete, reducing the information available for each decision. This makes the search behave more like Random during parallel batches, trading optimization quality for speed. More parallelism is not universally better (A). S3 costs are negligible compared to compute (C). Parallelism does not require GPUs (D).

---

### Question 11 -- T/F | Module 2 | Intermediate

**True or False: SageMaker's built-in BlazingText algorithm supports both Word2Vec (unsupervised word embedding) and text classification (supervised) modes within the same algorithm container.**

**Correct Answer: True**

**Rationale:** BlazingText implements both Word2Vec for learning word embeddings from unlabeled text and a supervised text classification mode based on the fastText architecture. The mode is selected via the `mode` hyperparameter: `batch_skipgram`, `skipgram`, or `cbow` for Word2Vec, and `supervised` for text classification. Both run in the same SageMaker container image.

---

### Question 12 -- Matching | Module 2 | Advanced

**Match each SageMaker built-in algorithm to the problem type it solves.**

| # | Algorithm | | Problem Type |
|---|-----------|---|-------------|
| 1 | XGBoost | | A) Anomaly detection on streaming or time-series data |
| 2 | K-Means | | B) Time-series forecasting with covariates |
| 3 | Random Cut Forest | | C) Supervised classification and regression on tabular data |
| 4 | DeepAR | | D) Unsupervised clustering for customer segmentation |
| 5 | BlazingText | | E) Text classification and word embedding generation |

**Correct Answers: 1-C, 2-D, 3-A, 4-B, 5-E**

**Rationale:** XGBoost is a gradient boosted tree algorithm for tabular supervised tasks (C). K-Means partitions data into k clusters (D). RCF identifies anomalous data points by measuring how much adding a point changes the tree structure (A). DeepAR is an autoregressive RNN for multi-series forecasting (B). BlazingText handles text classification and word embeddings (E).

---

### Question 13 -- MCQ | Module 3 | Intermediate | Scenario-Based

**A data scientist runs 5 training jobs with different hyperparameters but forgets to associate them with a SageMaker Experiment. She wants to compare their metrics side by side. What is the easiest retroactive fix?**

A) Delete all 5 training jobs and re-run them with an Experiment configured
B) Use the SageMaker SDK to retroactively associate the completed training jobs as Runs under a new Experiment, since training job metadata is preserved
C) Export CloudWatch metrics to a spreadsheet and compare manually
D) Re-run the jobs as an HPO tuning job instead

**Correct Answer: B**

**Rationale:** SageMaker preserves all training job metadata (parameters, metrics, artifacts) after completion. The SDK's `Run` or `Experiment` APIs allow associating existing training jobs with an Experiment after the fact. This avoids re-running jobs (A, D) and provides the console's built-in comparison views instead of manual spreadsheet work (C).

---

### Question 14 -- MCQ | Module 3 | Advanced | Scenario-Based

**A compliance auditor asks: "For model version 3 in your production registry, show me the exact dataset, algorithm, hyperparameters, and feature transformations used." Which SageMaker feature provides this end-to-end traceability?**

A) SageMaker Experiments alone
B) SageMaker Lineage Tracking, which records associations between Artifacts (datasets, models), Actions (training jobs, processing jobs), and Contexts (experiments, pipelines)
C) CloudTrail logs, which record every API call
D) The S3 bucket's versioning history

**Correct Answer: B**

**Rationale:** Lineage Tracking creates a provenance graph connecting all ML artifacts through typed entities: Artifacts (datasets, model files), Actions (training jobs, transform jobs), Contexts (experiments, pipeline executions), and Associations (directional edges between entities). This graph can trace from a deployed model back through its training job to the exact dataset version and processing pipeline. Experiments (A) track metrics and parameters but not the full artifact graph. CloudTrail (C) records API calls but not semantic ML relationships. S3 versioning (D) tracks file versions but not the processing logic.

---

### Question 15 -- MCQ | Module 3 | Advanced | Conceptual

**In SageMaker Lineage Tracking, what is the difference between an "Artifact" and an "Action"?**

A) Artifacts are AWS services; Actions are user operations
B) Artifacts represent data objects or model files (things that exist), while Actions represent compute operations like training jobs or processing jobs (things that happened)
C) Artifacts are immutable; Actions can be modified after creation
D) There is no meaningful difference; both terms are interchangeable

**Correct Answer: B**

**Rationale:** Lineage Tracking uses a typed entity model. Artifacts are tangible outputs -- datasets in S3, model files, Docker images. Actions are compute events -- training jobs, processing jobs, transform jobs. Associations connect them directionally (e.g., a training Action consumed a dataset Artifact and produced a model Artifact). They are not AWS services (A), and both are immutable once created (C). The distinction is fundamental, not superficial (D).

---

### Question 16 -- T/F | Module 3 | Intermediate

**True or False: SageMaker Lineage Tracking can show upstream provenance (tracing a model back to its training data) but cannot show downstream impact (tracing which endpoints use a specific model artifact).**

**Correct Answer: False**

**Rationale:** Lineage Tracking supports both upstream (what produced this artifact?) and downstream (what consumes this artifact?) queries. Associations are directional, enabling traversal in both directions. For example, you can query which endpoints currently serve a specific model version, or trace backward from an endpoint to the original training data.

---

### Question 17 -- MCQ | Module 3 | Expert | Scenario-Based

**A team needs to reproduce a model that was trained 6 months ago. They have the model artifact in S3 and the Experiment run metadata. However, they discover the Feature Store Feature Group used during training has been updated with new feature engineering logic. How should they approach reproducibility?**

A) Retrain with the current Feature Group; the new features are better anyway
B) Use the Offline Store's time-travel capability to query features as of the original training date, combined with Lineage Tracking to identify the exact feature versions used
C) Download the model artifact and skip retraining since the model already exists
D) Check the Git commit history for the feature engineering code

**Correct Answer: B**

**Rationale:** The Offline Store appends records with timestamps, enabling point-in-time queries that reconstruct the exact feature values available at training time. Combined with Lineage Tracking (which records which Feature Group version the training job consumed), the team can fully reproduce the training conditions. Simply retraining with current features (A) produces a different model. Downloading the existing artifact (C) does not reproduce the training process. Git history (D) shows code changes but not the actual feature data values at that point in time.

---

### Question 18 -- Matching | Module 3 | Advanced

**Match each tracking concept to the SageMaker feature that implements it.**

| # | Concept | | Feature |
|---|---------|---|---------|
| 1 | Comparing metrics across training runs | | A) Lineage Tracking |
| 2 | Tracing a model back to its training dataset | | B) Feature Store Offline Store |
| 3 | Point-in-time feature reconstruction | | C) SageMaker Experiments |
| 4 | Recording which pipeline step produced an artifact | | D) Lineage Tracking (Context + Association) |

**Correct Answers: 1-C, 2-A, 3-B, 4-D**

**Rationale:** Experiments provide comparison charts and metric tracking across runs (C). Lineage Tracking records provenance associations between artifacts and actions (A). The Offline Store preserves timestamped feature history for point-in-time queries (B). Pipeline Contexts in Lineage Tracking associate pipeline steps with their outputs (D).

---

### Question 19 -- MCQ | Module 4 | Intermediate | Scenario-Based

**A fraud detection model receives 5 requests per day on weekdays and zero on weekends. The current real-time endpoint on ml.m5.xlarge costs $0.23/hour (about $170/month). Which inference pattern would reduce costs the most?**

A) Batch Transform, running once per day
B) Serverless Inference, which scales to zero when idle and charges per invocation
C) Asynchronous Inference with a minimum instance count of 1
D) Multi-Model Endpoints with 3 model variants

**Correct Answer: B**

**Rationale:** Serverless Inference charges only for compute time during actual invocations, with no cost when idle. At 5 requests/day, the endpoint sits idle 99.99% of the time, making Serverless dramatically cheaper than a persistent real-time endpoint. Batch Transform (A) processes data in bulk but requires knowing when requests arrive. Async Inference (C) with a minimum instance of 1 still incurs persistent compute costs. Multi-Model Endpoints (D) share infrastructure across models but still run a persistent endpoint.

---

### Question 20 -- MCQ | Module 4 | Advanced | Scenario-Based

**A team deploys a Serverless Inference endpoint with MemorySizeInMB=2048 and MaxConcurrency=5. During a load test, they observe 8-second latency on the first invocation but consistent 200ms latency on subsequent invocations within the same minute. What explains the initial latency spike?**

A) The model artifact is too large for 2048 MB of memory
B) The cold start penalty: SageMaker must download the container image and model artifact, provision compute, and load the model into memory for the first invocation after idle
C) The MaxConcurrency setting of 5 is throttling the first request
D) The S3 bucket containing the model is in a different region

**Correct Answer: B**

**Rationale:** Serverless Inference scales to zero when idle. The first invocation triggers a cold start where SageMaker provisions compute, downloads the container and model, and loads the model into memory. Subsequent invocations within the keep-alive window reuse the warm container, resulting in consistent low latency. Memory size (A) would cause out-of-memory errors, not slow starts. MaxConcurrency (C) limits parallel invocations, not initial latency. Cross-region S3 (D) would add consistent latency to all invocations, not just the first.

---

### Question 21 -- MCQ | Module 4 | Advanced | Scenario-Based

**A SageMaker Asynchronous Inference endpoint is configured with an S3 output location and an SNS error topic. A client submits a request with a 50 MB payload. How does the invocation flow differ from real-time inference?**

A) There is no difference; the client receives the prediction synchronously
B) The client receives an immediate 202 response with an output location URL, the endpoint processes the payload asynchronously, writes the result to S3, and optionally sends an SNS notification on completion or failure
C) The client must poll the endpoint every second until the result is ready
D) SageMaker splits the 50 MB payload into smaller chunks and processes each as a separate real-time request

**Correct Answer: B**

**Rationale:** Async Inference decouples the request from the response. The client gets an immediate 202 Accepted with an S3 output location. SageMaker queues the request, processes it (supporting payloads up to 1 GB and processing times up to 15 minutes), and writes the result to S3. SNS notifications provide push-based completion signaling. Real-time inference (A) has a 60-second timeout and 6 MB payload limit. The client does not need to poll (C); SNS provides notification. SageMaker does not split payloads (D).

---

### Question 22 -- MCQ | Module 4 | Expert | Scenario-Based

**A company hosts 200 customer-specific fraud models. Each model is invoked 2-3 times per day. A Multi-Model Endpoint (MME) is deployed on a single ml.m5.xlarge instance. During peak hours, some invocations experience 5-second latency while others respond in 200ms. What is the most likely cause?**

A) The ml.m5.xlarge instance does not have enough vCPUs for 200 models
B) Model cache eviction: the instance cannot hold all 200 models in memory simultaneously, so less-frequently-used models are unloaded and must be re-loaded from S3 on invocation, causing latency spikes
C) The TargetModel parameter is being ignored by the endpoint
D) The endpoint needs auto-scaling enabled to handle peak traffic

**Correct Answer: B**

**Rationale:** MMEs dynamically load and unload models based on memory availability. With 200 models and limited instance memory, the endpoint evicts least-recently-used models. When an evicted model is invoked, it must be re-downloaded from S3 and re-loaded into memory, causing latency spikes. Frequently-invoked models stay warm in memory and respond quickly. vCPU count (A) affects throughput but not the observed cache-miss pattern. The TargetModel parameter (C) is required and honored by MMEs. Auto-scaling (D) adds instances but does not solve per-instance cache eviction.

---

### Question 23 -- MCQ | Module 4 | Advanced | Conceptual

**In a Multi-Container Endpoint with serial inference pipeline mode, Container A preprocesses input data and Container B runs the ML model. How does data flow between the containers?**

A) Container A writes to S3 and Container B reads from S3
B) Container A's response body is passed directly as Container B's request body through an internal network connection, with no S3 intermediate step
C) Both containers receive the original request simultaneously and their outputs are merged
D) Container A sends a message to an SQS queue that Container B consumes

**Correct Answer: B**

**Rationale:** In serial inference pipeline mode, SageMaker chains containers sequentially on the same endpoint. The output of Container A is passed directly to Container B's input through an internal connection, eliminating the latency of S3 round-trips. This in-memory passing enables low-latency preprocessing-to-prediction workflows. S3 (A) and SQS (D) would add unacceptable latency. Parallel execution with output merging (C) describes a different pattern not supported by serial pipelines.

---

### Question 24 -- T/F | Module 4 | Intermediate

**True or False: SageMaker Batch Transform is designed for one-time or scheduled bulk scoring of large datasets stored in S3, and it automatically provisions and terminates compute instances for the duration of the job.**

**Correct Answer: True**

**Rationale:** Batch Transform creates ephemeral compute instances, downloads the model and input data, processes all records, writes results to S3, and terminates the instances. There are no persistent endpoints and no ongoing compute charges after completion. This makes it ideal for bulk scoring scenarios where real-time response is unnecessary.

---

### Question 25 -- MCQ | Module 5 | Intermediate | Scenario-Based

**A team enables Data Capture on their fraud detection endpoint at 100% sampling. After one week, they notice the S3 data capture bucket has grown to 50 GB. What should they adjust to reduce storage costs while maintaining monitoring effectiveness?**

A) Disable Data Capture entirely
B) Reduce the sampling percentage to 10-20%, which captures a statistically representative sample while reducing storage by 80-90%
C) Switch to a larger S3 storage class
D) Delete the captured data daily

**Correct Answer: B**

**Rationale:** Data Capture sampling percentage controls what fraction of inference requests are captured. For drift detection, a statistically representative sample (10-20%) is sufficient and dramatically reduces storage costs. Disabling capture entirely (A) eliminates monitoring capability. A larger storage class (C) does not reduce volume. Daily deletion (D) removes the historical data needed for trend analysis.

---

### Question 26 -- MCQ | Module 5 | Advanced | Scenario-Based

**A Data Quality monitoring job detects that the `transaction_amount` feature has shifted: the training baseline shows a mean of $150 with standard deviation $80, but the captured inference data shows a mean of $420 with standard deviation $200. The monitoring report lists a K-S test violation. What does this indicate?**

A) The model is making incorrect predictions
B) The statistical distribution of the `transaction_amount` feature in production has significantly diverged from the training data distribution, which may cause model performance degradation
C) The K-S test is unreliable and should be replaced with a Chi-Squared test
D) The training data was incorrectly labeled

**Correct Answer: B**

**Rationale:** The K-S (Kolmogorov-Smirnov) test measures the maximum distance between two cumulative distribution functions. A violation means the production distribution has materially shifted from the baseline. This data drift is a leading indicator that the model may perform poorly since it was trained on different distribution characteristics. The K-S test does not directly measure prediction quality (A). K-S is appropriate for continuous features; Chi-Squared is for categorical (C). Data labels (D) are a separate concern from feature distribution drift.

---

### Question 27 -- MCQ | Module 5 | Advanced | Scenario-Based

**A Model Quality monitoring schedule compares production predictions against ground truth labels that arrive with a 48-hour delay. The latest report shows F1 score dropped from 0.92 (baseline) to 0.74. What type of drift does this indicate, and how does it differ from the data quality drift in the previous question?**

A) Data drift -- the input features have shifted
B) Concept drift -- the relationship between input features and the target variable has changed, causing the model to make worse predictions even though the input feature distributions may be stable
C) Label drift -- the ground truth labels are incorrect
D) Infrastructure drift -- the endpoint instance is degraded

**Correct Answer: B**

**Rationale:** Model Quality monitoring detects concept drift by comparing actual predictions against ground truth. When F1 drops despite potentially stable input distributions, it indicates the learned patterns no longer map correctly to outcomes (e.g., fraud tactics evolved). Data drift (A) is detected by Data Quality monitoring on input features, not prediction quality. Label drift (C) refers to target distribution changes, which is a subset of concept drift. Infrastructure drift (D) is not a standard ML monitoring category.

---

### Question 28 -- MCQ | Module 5 | Expert | Scenario-Based

**A production system uses Model Monitor with Data Quality, Model Quality, and Bias monitoring. The architect wants to automatically trigger retraining when any monitor detects violations. What AWS service chain implements this closed-loop architecture?**

A) Model Monitor -> CloudWatch Alarm -> SNS -> Email to team lead -> Manual retraining
B) Model Monitor -> CloudWatch Metric -> CloudWatch Alarm -> EventBridge Rule -> Lambda Function -> SageMaker Pipeline Start
C) Model Monitor -> S3 Event Notification -> EC2 Instance -> Python script
D) Model Monitor -> SageMaker Autopilot -> Automatic retraining

**Correct Answer: B**

**Rationale:** The closed-loop architecture chains: Model Monitor emits violation metrics to CloudWatch, a CloudWatch Alarm triggers when thresholds are breached, EventBridge captures the alarm state change and routes it to a Lambda function, and Lambda invokes a SageMaker Pipeline that retrains and re-evaluates the model. Option A requires human intervention (not automated). Option C uses unmanaged infrastructure. Model Monitor does not integrate directly with Autopilot (D).

---

### Question 29 -- T/F | Module 5 | Intermediate

**True or False: The K-S (Kolmogorov-Smirnov) test is used by SageMaker Model Monitor for detecting distribution drift in continuous numerical features, while the Chi-Squared test is used for detecting drift in categorical features.**

**Correct Answer: True**

**Rationale:** Model Monitor applies different statistical tests based on feature type. The K-S test compares cumulative distribution functions of continuous features. The Chi-Squared test compares observed vs. expected frequency distributions of categorical features. Using the wrong test type would produce unreliable results (e.g., K-S on categorical data with few categories lacks statistical power).

---

### Question 30 -- Matching | Module 5 | Advanced

**Match each monitoring type to what it detects.**

| # | Monitor Type | | What It Detects |
|---|-------------|---|-----------------|
| 1 | Data Quality Monitor | | A) Changes in prediction accuracy compared to ground truth |
| 2 | Model Quality Monitor | | B) Shifts in feature distributions compared to training baseline |
| 3 | Bias Monitor | | C) Changes in feature importance rankings over time |
| 4 | Feature Attribution Monitor | | D) Shifts in prediction fairness across protected groups |

**Correct Answers: 1-B, 2-A, 3-D, 4-C**

**Rationale:** Data Quality compares incoming feature distributions to baselines (B). Model Quality compares predictions to ground truth metrics (A). Bias Monitor uses Clarify to detect fairness shifts across demographic or protected groups (D). Feature Attribution Monitor tracks SHAP-based feature importance drift (C).

---

### Question 31 -- MCQ | Module 6 | Intermediate | Scenario-Based

**A training job on ml.m5.xlarge costs $0.23/hour and runs for 4 hours ($0.92 total). The same job configured with Managed Spot Training completes in 5 hours (including one interruption and restart) at the Spot rate of $0.069/hour ($0.345 total). What mandatory configuration enables the job to survive Spot interruptions?**

A) A larger instance type to train faster before interruption
B) An S3 checkpoint path where the training script periodically saves progress, allowing it to resume from the last checkpoint after a Spot interruption
C) Setting max_wait_time equal to max_run_time
D) Using multiple instances so training continues if one is interrupted

**Correct Answer: B**

**Rationale:** Managed Spot Training requires an S3 checkpoint path. The training script must periodically save model state and training progress to this path. When a Spot interruption occurs, SageMaker provisions a new Spot instance and resumes from the last checkpoint. Without checkpointing, the entire training job restarts from scratch after each interruption. Larger instances (A) do not prevent interruptions. max_wait_time should exceed max_run_time (C) to allow for Spot wait time. Multi-instance training (D) addresses distributed training, not Spot resilience.

---

### Question 32 -- MCQ | Module 6 | Advanced | Scenario-Based

**After a training job completes, CloudWatch shows average CPU utilization at 15% and memory utilization at 20% on ml.m5.xlarge (4 vCPUs, 16 GB RAM). What does this suggest?**

A) The job completed successfully so the metrics are irrelevant
B) The instance is significantly over-provisioned; the team should right-size to a smaller instance type like ml.m5.large (2 vCPUs, 8 GB) to reduce costs without impacting training time
C) The low utilization means the training script has a bug
D) CPU utilization should always be low; GPU utilization is what matters

**Correct Answer: B**

**Rationale:** Sustained low CPU and memory utilization indicates the instance has more resources than the workload requires. Right-sizing to a smaller instance (ml.m5.large at $0.115/hour vs. ml.m5.xlarge at $0.23/hour) can halve costs without degrading performance, since the workload only uses a fraction of the larger instance's capacity. Successful completion (A) does not mean resources were used efficiently. Low utilization is not a bug (C) but a sizing concern. For CPU-bound workloads (D), CPU utilization is the primary metric.

---

### Question 33 -- MCQ | Module 6 | Advanced | Scenario-Based

**A production endpoint serves fraud predictions with SLA requirements: p99 latency under 500ms, scale to handle 1000 TPS during peak. The team configures target tracking auto-scaling with InvocationsPerInstance=200 as the target metric. The endpoint starts with 2 instances. What happens when traffic reaches 800 TPS?**

A) The endpoint drops requests until traffic decreases
B) Auto-scaling adds instances to maintain approximately 200 invocations per instance, scaling to 4 instances (800/200=4), then continues monitoring and adjusting
C) Auto-scaling immediately provisions 10 instances as a safety buffer
D) The endpoint switches to Serverless Inference automatically

**Correct Answer: B**

**Rationale:** Target tracking auto-scaling adjusts instance count to keep the per-instance metric near the target value. At 800 TPS with a target of 200 invocations/instance, the policy scales to 4 instances. As traffic changes, the policy continuously adjusts (scale-out and scale-in). It does not drop requests (A) unless the scaling cannot keep pace. It scales proportionally, not by fixed large increments (C). There is no automatic switch between inference modes (D).

---

### Question 34 -- MCQ | Module 6 | Expert | Scenario-Based

**An ML platform team needs to ensure that training data never traverses the public internet. Training jobs read from S3 and write model artifacts back to S3. The SageMaker API calls must also stay within the private network. Which combination of services achieves this?**

A) Configure the training job to run in a private VPC subnet with no internet gateway, create a VPC Endpoint (Gateway) for S3, and create VPC Endpoints (Interface/PrivateLink) for the SageMaker API and SageMaker Runtime API
B) Use S3 Transfer Acceleration to speed up private transfers
C) Enable server-side encryption on S3 buckets
D) Deploy the training job in the default VPC with a NAT gateway

**Correct Answer: A**

**Rationale:** Running in a private subnet (no internet gateway) ensures no public internet path exists. The S3 Gateway Endpoint enables private S3 access without internet. Interface Endpoints (powered by PrivateLink) for SageMaker API and Runtime ensure API calls stay within the AWS private network. Transfer Acceleration (B) still uses public endpoints. Encryption (C) protects data at rest/in transit but does not restrict network paths. A NAT gateway (D) routes traffic through the public internet.

---

### Question 35 -- T/F | Module 6 | Intermediate

**True or False: SageMaker Inference Recommender runs load tests against your model on multiple instance types and returns a recommendation ranked by latency, throughput, and cost, helping you choose the right instance without manual benchmarking.**

**Correct Answer: True**

**Rationale:** Inference Recommender offers Default and Advanced job types. The Default job tests your model on a curated set of instance types and provides recommendations ranked by performance and cost metrics. The Advanced job lets you specify custom instance types and traffic patterns. Both eliminate the need for manual benchmarking across instance types.

---

### Question 36 -- MCQ | Module 6 | Expert | Scenario-Based

**A training job processes sensitive healthcare data. The security team requires: (1) data encrypted at rest in S3, (2) data encrypted on the training instance's storage volume, and (3) all API calls audited. Which combination of AWS services satisfies all three requirements?**

A) S3 default encryption, EBS encryption on the training instance, and CloudTrail for API auditing
B) S3 SSE-KMS with a customer-managed KMS key for S3 encryption, VolumeKmsKeyId parameter in the training job configuration for EBS volume encryption with the same or separate KMS key, and CloudTrail enabled for SageMaker API call auditing
C) S3 bucket policies, SageMaker notebook encryption, and CloudWatch Logs
D) VPC PrivateLink for all three requirements

**Correct Answer: B**

**Rationale:** SSE-KMS with customer-managed keys provides auditable, rotatable encryption for S3 objects. The VolumeKmsKeyId parameter in the training job configuration encrypts the instance's EBS storage volume with a KMS key, protecting data at rest on the training instance. CloudTrail records all SageMaker API calls for audit purposes. Default encryption (A) uses AWS-managed keys that lack customer control. Bucket policies (C) control access, not encryption. PrivateLink (D) controls network paths, not encryption or auditing.

---
