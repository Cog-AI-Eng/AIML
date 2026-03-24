# AppliedML-Foundations Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | ML Lifecycle in Action: Data Ingestion Through Preprocessing | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Supervised Models: Linear Regression and Logistic Regression | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Algorithm Selection, Unsupervised Exploration, and Git Workflow | 45 |
| Buffer | Open Q&A, Git Branch Activity, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- TalentFlow HR Analytics**

Associates will play the role of a data scientist at a fictional company called TalentFlow, which is experiencing high employee turnover. Leadership has asked two questions:

1. **"Can we predict an employee's monthly income based on their profile?"** (Regression -- continuous target)
2. **"Can we predict which employees are likely to leave?"** (Classification -- binary target)

This single scenario threads through every exit criterion: Associates will walk the ML lifecycle end-to-end, enforce reproducibility, distinguish supervised from unsupervised paradigms, build both Linear and Logistic Regression models, and apply the algorithm selection framework to justify their choices. A brief unsupervised clustering segment shows how the same data supports a different paradigm when labels are removed.

**Why a synthetic dataset?** The lecture generates data inline with NumPy so there are zero download dependencies, every student sees identical numbers, and the instructor controls exactly which patterns appear. This also reinforces the reproducibility lesson because Associates can verify that their outputs match the instructor's.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Python 3.10+ environment verified (`python --version`)
- [ ] Required packages installed: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`
- [ ] Jupyter Notebook server running or VS Code with Jupyter extension ready
- [ ] Terminal visible for Git commands in the final activity
- [ ] Font size increased for projector readability (16pt+ in notebook, 14pt+ in terminal)
- [ ] A clean working directory: `mkdir talentflow-ml && cd talentflow-ml && git init`

### Quick Install (if needed)

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Recommended Notebook Name

`talentflow_foundations.ipynb`

---

## Stage 1: The ML Lifecycle in Action -- Data Ingestion Through Preprocessing

**Duration:** 45 minutes
**Exit Criteria Addressed:** Explain the standard ML lifecycle steps from data ingestion to deployment; Demonstrate code reproducibility strategies using random seeds and version control

### Instructor Opening (5 minutes -- talk, no code)

> "You have read about the ML lifecycle as a sequence of stages: problem framing, data ingestion, EDA, feature engineering, model training, evaluation, and deployment. Today we are going to live that sequence, not just describe it. By the end of this session you will have built two models from scratch on the same dataset, compared them against baselines, and committed your work to Git. Everything we do will be reproducible -- same seeds, same splits, same results on your machine as on mine."

Briefly sketch the lifecycle stages on a whiteboard or shared diagram. Point at each stage and say which part of the lecture will cover it:

- **Problem Framing** -- right now (the TalentFlow scenario)
- **Data Ingestion** -- Stage 1, first code cell
- **EDA and Cleaning** -- Stage 1
- **Feature Engineering and Preprocessing** -- Stage 1
- **Model Training and Evaluation** -- Stages 2 and 3
- **Deployment concept** -- Stage 3 closing discussion (not coded, but discussed)

---

### STEP 1 -- Reproducibility Setup (5 minutes)

**Pacing: line-by-line.** Type each import and seed statement individually. Explain why each seed matters.

```python
# STEP 1: Reproducibility setup
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Fix all sources of randomness BEFORE any data generation or splitting
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"Random seed locked to {SEED}")
print(f"NumPy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
```

**Instructor Note:** Pause after running this cell. Ask: *"Why do we print the versions?"* Answer: so that if results differ later, we can check whether a library update changed behavior. This is a lightweight reproducibility habit.

---

### STEP 2 -- Data Ingestion: Generating the TalentFlow Dataset (10 minutes)

**Pacing: block update.** Paste the full generation block, then walk through it section by section after execution. Associates should see the output first, then understand how it was built.

```python
# STEP 2: Data ingestion -- generate synthetic TalentFlow HR dataset
n_employees = 600

data = {
    "employee_id": np.arange(1, n_employees + 1),
    "age": np.random.randint(22, 62, size=n_employees),
    "years_at_company": np.random.randint(0, 30, size=n_employees),
    "distance_from_home": np.random.randint(1, 50, size=n_employees),
    "satisfaction_score": np.random.uniform(1.0, 5.0, size=n_employees).round(2),
    "monthly_hours": np.random.normal(loc=170, scale=20, size=n_employees).round(1),
    "num_projects": np.random.randint(1, 8, size=n_employees),
    "department": np.random.choice(
        ["Engineering", "Sales", "HR", "Marketing"], size=n_employees, p=[0.4, 0.3, 0.15, 0.15]
    ),
    "overtime": np.random.choice([0, 1], size=n_employees, p=[0.6, 0.4]),
}

base_income = (
    25000
    + data["age"] * 300
    + data["years_at_company"] * 500
    + data["num_projects"] * 400
    + np.where(np.array(data["department"]) == "Engineering", 5000, 0)
    + np.random.normal(0, 2000, size=n_employees)
)
data["monthly_income"] = np.clip(base_income, 20000, 80000).round(2)

attrition_score = (
    -0.5 * data["satisfaction_score"]
    + 0.03 * data["distance_from_home"]
    + 0.8 * data["overtime"]
    - 0.02 * data["years_at_company"]
    + 0.01 * data["monthly_hours"]
    + np.random.normal(0, 0.3, size=n_employees)
)
attrition_prob = 1 / (1 + np.exp(-attrition_score))
data["attrition"] = (attrition_prob > 0.55).astype(int)

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Attrition rate: {df['attrition'].mean():.2%}")
df.head(10)
```

**Instructor Note:** After running, highlight these points:

- The income formula is intentionally linear with noise -- this is why Linear Regression will work well here.
- The attrition labels are generated through a logistic function -- this is why Logistic Regression is a natural fit.
- Both relationships are *designed* so Associates see strong results, building confidence before they face messy real-world data.
- Point out the class balance in attrition. If it is imbalanced, discuss why accuracy alone would be misleading.

> "In a real project, this step would be loading CSVs, querying a database, or pulling from an API. The lifecycle stage is the same: bring data into a controlled environment where you can inspect it."

---

### STEP 3 -- Exploratory Data Analysis (12 minutes)

**Pacing: line-by-line for the first two plots, then block update for the remainder.** Narrate each visualization choice.

```python
# STEP 3a: Basic profiling
print(df.info())
print("\n--- Descriptive Statistics ---")
df.describe()
```

```python
# STEP 3b: Distribution of the continuous target (monthly_income)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["monthly_income"], bins=30, edgecolor="black", alpha=0.7)
axes[0].set_title("Monthly Income Distribution")
axes[0].set_xlabel("Income")
axes[0].set_ylabel("Count")

sns.boxplot(x="department", y="monthly_income", data=df, ax=axes[1])
axes[1].set_title("Income by Department")

plt.tight_layout()
plt.show()
```

**Instructor Note:** Ask Associates: *"What do you notice about Engineering versus the other departments?"* The synthetic data has a 5000 bump for Engineering. This is a feature engineering insight they will use later.

```python
# STEP 3c: Distribution of the binary target (attrition)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["attrition"].value_counts().plot.bar(ax=axes[0], edgecolor="black", alpha=0.7)
axes[0].set_title("Attrition Counts (0 = Stayed, 1 = Left)")
axes[0].set_xlabel("Attrition")
axes[0].set_ylabel("Count")

sns.boxplot(x="attrition", y="satisfaction_score", data=df, ax=axes[1])
axes[1].set_title("Satisfaction by Attrition Status")

plt.tight_layout()
plt.show()
```

```python
# STEP 3d: Correlation heatmap (numeric columns only)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "employee_id"]

plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

**Instructor Note:** Walk the heatmap row by row. Highlight correlations with `monthly_income` (the regression target) and `attrition` (the classification target). Mention that strong correlations with the target are good signals for linear models; correlations among features can cause multicollinearity but are acceptable for this baseline demonstration.

---

### STEP 4 -- Feature Engineering and Preprocessing (13 minutes)

**Pacing: line-by-line.** This is the most important section for data leakage prevention. Slow down and emphasize the split-before-scale rule.

```python
# STEP 4a: Separate features and targets
from sklearn.model_selection import train_test_split

feature_cols = [
    "age", "years_at_company", "distance_from_home",
    "satisfaction_score", "monthly_hours", "num_projects", "overtime"
]

X = df[feature_cols].copy()
y_income = df["monthly_income"].copy()
y_attrition = df["attrition"].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Income target shape:  {y_income.shape}")
print(f"Attrition target shape: {y_attrition.shape}")
```

**Instructor Note:** Point out that `department` is excluded for now. In a follow-up discussion, mention that one-hot encoding would bring it in, but keeping numeric-only features simplifies the first pass and keeps focus on the lifecycle.

```python
# STEP 4b: Train-test split -- BEFORE any scaling
# The same split is used for BOTH tasks so results are comparable
X_train, X_test, y_inc_train, y_inc_test, y_att_train, y_att_test = train_test_split(
    X, y_income, y_attrition,
    test_size=0.2,
    random_state=SEED,
    stratify=y_attrition
)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set:     {X_test.shape[0]} rows")
print(f"Train attrition rate: {y_att_train.mean():.2%}")
print(f"Test  attrition rate: {y_att_test.mean():.2%}")
```

> **CRITICAL TEACHING MOMENT:** "We split FIRST, then scale. If we scaled the entire dataset before splitting, information from the test set would leak into the training set through the mean and standard deviation. This is data leakage. It makes your metrics look better than they really are, and your model will underperform in production."

```python
# STEP 4c: Scale numeric features AFTER splitting
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaler fit on training data only.")
print(f"Train means (should be ~0): {X_train_scaled.mean(axis=0).round(4)}")
print(f"Test means  (will NOT be 0): {X_test_scaled.mean(axis=0).round(4)}")
```

**Instructor Note:** Emphasize that `.fit_transform()` is called on training data and `.transform()` (without fit) is called on test data. The test means will NOT be exactly zero, and that is correct -- the scaler learned its parameters from training data only.

[PAUSE FOR Q&A] -- 3 minutes

> "Before we move on: does anyone have questions about the lifecycle stages we have covered so far, the reproducibility setup, or the split-before-scale rule?"

---

[PAUSE FOR BREAK] -- 10 minutes

> "We have completed the first half of the lifecycle: problem framing, data ingestion, EDA, feature engineering, and preprocessing. When we come back, we train models."

---

## Stage 2: Building Supervised Models -- Linear Regression and Logistic Regression

**Duration:** 45 minutes
**Exit Criteria Addressed:** Differentiate between supervised, unsupervised, and reinforcement learning paradigms; Build a Linear Regression model to predict continuous target variables; Build a Logistic Regression model for binary classification tasks

### Instructor Bridge (3 minutes -- talk, no code)

> "We are now at the Model Training and Evaluation stages of the lifecycle. Remember that supervised learning needs both X and y -- the model learns from labeled examples. We have two supervised tasks: predicting income (regression) and predicting attrition (classification). We will tackle regression first."

---

### STEP 5 -- Linear Regression: Predicting Monthly Income (15 minutes)

**Pacing: line-by-line for model creation and fitting, block update for metrics.**

```python
# STEP 5a: Train a Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_inc_train)

y_inc_pred = lr_model.predict(X_test_scaled)

print("--- Linear Regression: Income Prediction ---")
print(f"R-squared:           {r2_score(y_inc_test, y_inc_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_inc_test, y_inc_pred):.2f}")
print(f"Root Mean Sq Error:  {np.sqrt(mean_squared_error(y_inc_test, y_inc_pred)):.2f}")
```

**Instructor Note:** Explain each metric:
- **R-squared** -- proportion of variance explained; 1.0 is perfect, 0.0 means no better than predicting the mean.
- **MAE** -- average dollar error; easy for stakeholders to understand ("on average we are off by $X").
- **RMSE** -- penalizes large errors more than MAE.

> "Notice we report three metrics, not just one. The readings warned against relying on a single number."

```python
# STEP 5b: Inspect model coefficients
coeff_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coef_
}).sort_values("coefficient", ascending=False)

print("\nFeature Coefficients (standardized scale):")
print(coeff_df.to_string(index=False))
print(f"\nIntercept: {lr_model.intercept_:.2f}")
```

**Instructor Note:** Since features are standardized, coefficient magnitudes are directly comparable. Walk through which features have the strongest positive and negative effects. Connect back to the data generation formula -- Associates should see that `years_at_company`, `age`, and `num_projects` have large positive coefficients, matching how we built the data.

```python
# STEP 5c: Residual analysis
residuals = y_inc_test - y_inc_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_inc_pred, residuals, alpha=0.5, edgecolors="black", linewidth=0.5)
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Income")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=25, edgecolor="black", alpha=0.7)
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual")

plt.tight_layout()
plt.show()
```

> "If the residuals look randomly scattered around zero with no funnel shape, our linear assumption is reasonable. If the histogram is roughly bell-shaped, the noise is well-behaved. These are not formal tests, but they are good diagnostic habits."

[PAUSE FOR Q&A] -- 2 minutes

---

### STEP 6 -- Logistic Regression: Predicting Attrition (18 minutes)

**Pacing: line-by-line for model creation and fitting. Block update for the classification report and confusion matrix.**

```python
# STEP 6a: Train a Logistic Regression model
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(random_state=SEED, max_iter=1000)
log_model.fit(X_train_scaled, y_att_train)

y_att_pred = log_model.predict(X_test_scaled)
y_att_proba = log_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression trained.")
print(f"Classes: {log_model.classes_}")
```

**Instructor Note:** Point out `max_iter=1000` -- the default (100) can cause convergence warnings on some datasets. Also note `random_state=SEED` for reproducibility of the solver's internal randomness.

```python
# STEP 6b: Classification metrics -- NOT just accuracy
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("--- Logistic Regression: Attrition Prediction ---")
print(f"Accuracy:  {accuracy_score(y_att_test, y_att_pred):.4f}")
print(f"Precision: {precision_score(y_att_test, y_att_pred):.4f}")
print(f"Recall:    {recall_score(y_att_test, y_att_pred):.4f}")
print(f"F1 Score:  {f1_score(y_att_test, y_att_pred):.4f}")
print("\nFull Classification Report:")
print(classification_report(y_att_test, y_att_pred, target_names=["Stayed", "Left"]))
```

> **CRITICAL TEACHING MOMENT:** "Accuracy alone is banned in this course, and here is why. If 70% of employees stay, a model that always predicts 'Stayed' gets 70% accuracy while catching zero attrition cases. That is useless for HR. Precision tells us: of everyone we flagged as leaving, how many actually left? Recall tells us: of everyone who actually left, how many did we catch? F1 balances the two."

```python
# STEP 6c: Confusion matrix visualization
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    log_model, X_test_scaled, y_att_test,
    display_labels=["Stayed", "Left"],
    cmap="Blues",
    ax=ax
)
ax.set_title("Confusion Matrix -- Logistic Regression")
plt.tight_layout()
plt.show()
```

**Instructor Note:** Walk through each quadrant of the confusion matrix:
- **Top-left (TN):** Correctly predicted "Stayed"
- **Top-right (FP):** Predicted "Left" but they stayed (false alarm)
- **Bottom-left (FN):** Predicted "Stayed" but they left (missed attrition -- the costly error for HR)
- **Bottom-right (TP):** Correctly predicted "Left"

Ask: *"If you are an HR director, which error costs you more: a false alarm or a missed departure?"* Use the answer to motivate recall as the priority metric in this scenario.

```python
# STEP 6d: Probability distribution for predicted attrition
fig, ax = plt.subplots(figsize=(8, 4))
for label, name in [(0, "Stayed"), (1, "Left")]:
    mask = y_att_test == label
    ax.hist(y_att_proba[mask], bins=20, alpha=0.6, label=name, edgecolor="black")

ax.set_xlabel("Predicted Probability of Leaving")
ax.set_ylabel("Count")
ax.set_title("Predicted Probabilities by Actual Outcome")
ax.legend()
ax.axvline(x=0.5, color="red", linestyle="--", label="Default Threshold")
plt.tight_layout()
plt.show()
```

> "This histogram shows us how confident the model is. If the two distributions overlap heavily near 0.5, the model is uncertain. If they separate cleanly, the model has learned a strong signal. This is richer information than a single accuracy number."

### Supervised vs. Unsupervised Recap (5 minutes -- talk, minimal code)

> "Let us step back. Everything we just did was supervised learning. We gave the model X and y. The model optimized a loss function that compared its predictions to known answers. Linear Regression minimized squared error against numeric income labels. Logistic Regression maximized the likelihood of binary attrition labels. In both cases, we had an answer key."
>
> "In Stage 3, we will briefly remove the labels and ask: can we find natural groupings in this workforce without being told who left and who stayed? That is unsupervised learning. And we will also build baselines to see whether our models actually beat trivial strategies. That is the algorithm selection framework."

[PAUSE FOR Q&A] -- 2 minutes

---

[PAUSE FOR BREAK] -- 10 minutes

> "Take a full break. When we come back, we put our models on trial: are they actually better than guessing?"

---

## Stage 3: Algorithm Selection, Unsupervised Exploration, and Git Workflow

**Duration:** 45 minutes
**Exit Criteria Addressed:** Apply an algorithm selection framework to choose base models for tabular data; Differentiate between supervised, unsupervised, and reinforcement learning paradigms; Demonstrate code reproducibility strategies using random seeds and version control

### Instructor Bridge (2 minutes -- talk, no code)

> "The algorithm selection framework says: never declare a model good until you have compared it against baselines. A DummyClassifier that always predicts the majority class is the floor. If your real model cannot beat the floor, something is wrong with your data, your features, or your evaluation -- not your choice of algorithm."

---

### STEP 7 -- Baseline Comparison: DummyClassifier (10 minutes)

**Pacing: line-by-line.** This is a pivotal concept. Let Associates absorb each comparison.

```python
# STEP 7a: Build a DummyClassifier baseline
from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy_model.fit(X_train_scaled, y_att_train)

y_dummy_pred = dummy_model.predict(X_test_scaled)

print("--- Dummy Classifier (Always Predicts Majority Class) ---")
print(f"Accuracy:  {accuracy_score(y_att_test, y_dummy_pred):.4f}")
print(f"Precision: {precision_score(y_att_test, y_dummy_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_att_test, y_dummy_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_att_test, y_dummy_pred, zero_division=0):.4f}")
```

**Instructor Note:** The DummyClassifier will likely have decent accuracy (because it rides the majority class) but zero recall for the minority class. This is the clearest possible demonstration of why accuracy alone is dangerous.

```python
# STEP 7b: Side-by-side comparison
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1"],
    "DummyClassifier": [
        accuracy_score(y_att_test, y_dummy_pred),
        precision_score(y_att_test, y_dummy_pred, zero_division=0),
        recall_score(y_att_test, y_dummy_pred, zero_division=0),
        f1_score(y_att_test, y_dummy_pred, zero_division=0),
    ],
    "LogisticRegression": [
        accuracy_score(y_att_test, y_att_pred),
        precision_score(y_att_test, y_att_pred),
        recall_score(y_att_test, y_att_pred),
        f1_score(y_att_test, y_att_pred),
    ],
})

print("--- Model Comparison ---")
print(comparison.to_string(index=False))
```

> "Look at the Recall and F1 columns. The Dummy model catches nobody who leaves. Logistic Regression is imperfect, but it actually identifies at-risk employees. This is the algorithm selection framework in action: we did not guess that logistic regression was good -- we measured it against a principled baseline."

```python
# STEP 7c: Visualize the comparison
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(comparison["Metric"]))
width = 0.35

bars1 = ax.bar(x_pos - width / 2, comparison["DummyClassifier"], width, label="DummyClassifier", alpha=0.7)
bars2 = ax.bar(x_pos + width / 2, comparison["LogisticRegression"], width, label="LogisticRegression", alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels(comparison["Metric"])
ax.set_ylabel("Score")
ax.set_title("Baseline vs Logistic Regression")
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.show()
```

---

### STEP 8 -- Algorithm Selection Framework Walkthrough (8 minutes)

**Pacing: talk through the framework, type the summary as a markdown cell or printed output.** This is a conceptual consolidation step.

```python
# STEP 8: Document the algorithm selection reasoning
framework_summary = """
=== ALGORITHM SELECTION FRAMEWORK -- TalentFlow Attrition ===

1. TASK TYPE
   - Supervised binary classification (predict attrition: 0 or 1)
   - Supervised regression (predict monthly income: continuous)

2. DATA DESCRIPTION
   - 600 rows, 7 numeric features (tabular)
   - No missing values, moderate noise
   - Relationships are approximately linear by construction
   - Binary target has mild class imbalance

3. CONSTRAINTS
   - Interpretability required (HR stakeholders need explanations)
   - Low latency not critical (batch predictions, not real-time)
   - Small team, so model maintenance cost matters

4. BASELINES
   - DummyClassifier (most_frequent): floor for classification
   - LinearRegression: baseline for regression
   - LogisticRegression: baseline for classification

5. RESULTS AND DECISION
   - DummyClassifier: high accuracy, zero recall (useless for the business goal)
   - LogisticRegression: strong improvement in recall and F1
   - LinearRegression: reasonable R-squared with interpretable coefficients
   - DECISION: LogisticRegression is justified for classification;
     LinearRegression is justified for regression.
   - NEXT STEP (future modules): try tree-based ensembles if linear
     models plateau on more complex data.
"""
print(framework_summary)
```

> "This is the framework from your readings, applied to a real decision. Task type, data description, constraints, baselines, comparison. If you can produce a block like this for any dataset, you are applying the framework, not just memorizing algorithm names."

[PAUSE FOR Q&A] -- 3 minutes

---

### STEP 9 -- Unsupervised Learning: K-Means Clustering (12 minutes)

**Pacing: line-by-line for the clustering setup, block update for the visualization.**

> "Now we shift paradigms. We are going to pretend we do not have the attrition labels. Can we find natural groupings in the employee data? This is unsupervised learning -- no y, just X."

```python
# STEP 9a: K-Means clustering on the same features (no labels)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
kmeans.fit(X_train_scaled)

cluster_labels = kmeans.predict(X_train_scaled)
print(f"Cluster sizes: {np.bincount(cluster_labels)}")
```

**Instructor Note:** Emphasize: `fit()` receives only X, not y. The model discovers structure without supervision.

```python
# STEP 9b: Visualize clusters using the two most informative features
train_df = X_train.copy()
train_df["cluster"] = cluster_labels
train_df["actual_attrition"] = y_att_train.values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter1 = axes[0].scatter(
    train_df["satisfaction_score"], train_df["monthly_hours"],
    c=train_df["cluster"], cmap="viridis", alpha=0.6, edgecolors="black", linewidth=0.3
)
axes[0].set_xlabel("Satisfaction Score")
axes[0].set_ylabel("Monthly Hours")
axes[0].set_title("K-Means Clusters (Unsupervised)")
plt.colorbar(scatter1, ax=axes[0], label="Cluster")

scatter2 = axes[1].scatter(
    train_df["satisfaction_score"], train_df["monthly_hours"],
    c=train_df["actual_attrition"], cmap="coolwarm", alpha=0.6, edgecolors="black", linewidth=0.3
)
axes[1].set_xlabel("Satisfaction Score")
axes[1].set_ylabel("Monthly Hours")
axes[1].set_title("Actual Attrition Labels (Supervised Ground Truth)")
plt.colorbar(scatter2, ax=axes[1], label="Attrition")

plt.tight_layout()
plt.show()
```

> "The left plot shows clusters the algorithm found on its own. The right plot shows the actual attrition labels. Do the clusters align with attrition? Sometimes they do, sometimes they do not. The point is that unsupervised learning finds *structure*, not necessarily the structure you care about. That is why, when you have labels and a prediction goal, supervised learning is the right paradigm."

```python
# STEP 9c: Cross-tabulate clusters with actual attrition
crosstab = pd.crosstab(
    train_df["cluster"], train_df["actual_attrition"],
    rownames=["Cluster"], colnames=["Attrition"]
)
print("Cluster vs Actual Attrition:")
print(crosstab)
print()

for cluster_id in sorted(train_df["cluster"].unique()):
    cluster_mask = train_df["cluster"] == cluster_id
    att_rate = train_df.loc[cluster_mask, "actual_attrition"].mean()
    print(f"Cluster {cluster_id}: {cluster_mask.sum()} employees, attrition rate = {att_rate:.2%}")
```

**Instructor Note:** If one cluster has a notably higher attrition rate, point it out: *"This cluster might represent a high-risk segment. In practice, you could use unsupervised insights to inform feature engineering for supervised models. The paradigms complement each other."*

### Reinforcement Learning -- Brief Contrast (2 minutes -- talk only)

> "We have now seen supervised and unsupervised learning in code. The third paradigm is reinforcement learning. We will not code it today, but the mental model is simple: an agent takes actions in an environment, receives rewards or penalties, and learns a policy over time. Think of it as learning to play a game through trial and error. The key difference: there is no static dataset of labeled rows. The data is generated by the agent's interaction with the world. For tabular business problems like TalentFlow, supervised and unsupervised are the relevant paradigms."

---

### STEP 10 -- Git Workflow and Version Control (8 minutes)

**Pacing: line-by-line in the terminal.** Switch from the notebook to a visible terminal window.

> "The last piece of reproducibility is version control. We have a notebook that generates consistent results because of our seed. Now we commit it so anyone on the team can trace exactly what code produced these results."

**In the terminal:**

```bash
# STEP 10a: Verify we are in the project directory with a git repo
cd talentflow-ml
git status
```

```bash
# STEP 10b: Create a meaningful .gitignore
echo "__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store" > .gitignore
```

```bash
# STEP 10c: Stage and commit the initial work
git add .gitignore talentflow_foundations.ipynb
git commit -m "Add foundations notebook: EDA, Linear Reg, Logistic Reg, KMeans baseline"
```

```bash
# STEP 10d: Create an experiment branch
git checkout -b experiment/add-polynomial-features
```

> "This branch is where you would try adding polynomial features or a different scaler without risking the working baseline on main. When the experiment works, you merge it back. When it fails, you delete the branch and your main branch is untouched."

```bash
# STEP 10e: Show branch list
git branch
```

```bash
# STEP 10f: Switch back to main
git checkout main
```

**Instructor Note:** If time allows, have Associates create their own branch named `experiment/<their-initials>-feature-test` and commit a small change (for example, changing the number of KMeans clusters to 4). This is the Git branch activity.

---

## Git Branch Activity (Remaining Buffer Time)

**Instructions to give Associates verbally or in a shared message:**

> "Here is your mini-activity for the remaining time. Do the following in order:"

1. **Create a new branch** from main:
   ```bash
   git checkout -b experiment/your-initials-kmeans-test
   ```

2. **Open the notebook** and change the number of K-Means clusters from 3 to 4. Rerun the clustering cell and the cross-tabulation cell.

3. **Save the notebook** and commit:
   ```bash
   git add talentflow_foundations.ipynb
   git commit -m "Experiment: test 4 clusters in KMeans segmentation"
   ```

4. **Switch back to main** and verify your original 3-cluster results are preserved:
   ```bash
   git checkout main
   ```

5. **Verify** by opening the notebook on main -- the cluster count should still be 3.

> "This exercise proves that branches isolate experiments. Your baseline is safe on main while you try ideas on feature branches. This is how reproducibility and version control work together in practice."

---

[PAUSE FOR Q&A] -- Remaining time

> "We have walked the entire ML lifecycle from raw data to committed, version-controlled models. Let us use the remaining time for any questions about what we covered."

---

## Wrap-Up: Exit Criteria Checklist

Use this checklist to verify all required exit criteria were demonstrated during the lecture.

| Exit Criterion | Where Demonstrated |
|---|---|
| Explain the standard ML lifecycle steps from data ingestion to deployment | Stage 1 opening discussion; lifecycle stages mapped to lecture sections; deployment discussed conceptually in Stage 3 |
| Demonstrate code reproducibility strategies using random seeds and version control | STEP 1 (seed setup), STEP 4b (random_state in split), STEP 10 (Git workflow and branching) |
| Differentiate between supervised, unsupervised, and reinforcement learning paradigms | Stage 2 recap discussion; STEP 9 (K-Means unsupervised); Stage 3 RL verbal contrast |
| Build a Linear Regression model to predict continuous target variables | STEP 5 (full pipeline: fit, predict, evaluate with R2/MAE/RMSE, coefficient inspection, residual plot) |
| Build a Logistic Regression model for binary classification tasks | STEP 6 (full pipeline: fit, predict, predict_proba, confusion matrix, precision/recall/F1) |
| Apply an algorithm selection framework to choose base models for tabular data | STEP 7 (DummyClassifier baseline), STEP 8 (framework documentation with all five decision points) |

---

## Instructor Debrief Notes

**Common student questions and suggested responses:**

- *"Why did we not include the department column?"* -- Encoding categorical variables is a valid next step. The decision to start numeric-only was intentional: it keeps the first pass simple and gives Associates a clear upgrade path for the assignment.

- *"Should we always use StandardScaler?"* -- No. MinMaxScaler, RobustScaler, and no scaling at all are valid depending on the algorithm and data. Linear models benefit from scaling; tree-based models generally do not need it. This is an algorithm selection consideration.

- *"Is 42 a magic number?"* -- No. Any fixed integer works. The number 42 is a community convention (a reference to The Hitchhiker's Guide to the Galaxy). What matters is that the seed is documented and consistent across related experiments.

- *"When would unsupervised learning be the primary approach instead of a supplement?"* -- When there are no labels at all (customer segmentation without predefined categories), when the goal is dimensionality reduction for visualization, or when the task is anomaly detection without labeled anomalies.

**Post-lecture follow-up:**

- Direct Associates to the AppliedML-Foundations Assignment where they will apply these techniques to a different dataset independently.
- Remind Associates that future modules (Evaluation, Neural Networks, Deep Learning, Transformers) build on this foundation -- the lifecycle, reproducibility habits, and baseline-first thinking carry forward into every module.
