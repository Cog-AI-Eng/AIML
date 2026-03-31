# Week 1 -- Interview-Ready Cheat Sheet

> Covers Tuesday through Friday: ML Foundations, Evaluation, Neural Networks, SageMaker, and Deep Learning Architectures.
> Every concept follows the same template: **What** / **Why** / **How** / **Where** / **Gotcha**.

---

## Table of Contents

- [Tuesday -- Applied ML Foundations](#tuesday----applied-ml-foundations)
- [Wednesday -- Applied ML Evaluation](#wednesday----applied-ml-evaluation)
- [Thursday -- Neural Networks and Advanced Evaluation](#thursday----neural-networks-and-advanced-evaluation)
- [Friday -- SageMaker Foundations, Training, and Deep Learning](#friday----sagemaker-foundations-training-and-deep-learning)

---

# Tuesday -- Applied ML Foundations

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | ML Lifecycle | The seven-stage journey from raw data to a running, improvable system |
| 2 | Reproducibility | Same inputs, same outputs -- every time, every teammate |
| 3 | Supervised Learning | Learn from labeled examples to predict new outcomes |
| 4 | Unsupervised Learning | Find hidden structure when there are no labels |
| 5 | Reinforcement Learning | An agent learns by trial, error, and rewards |
| 6 | Classification vs Regression | Predicting a category vs predicting a number |
| 7 | Linear Regression | The baseline model for continuous targets |
| 8 | Logistic Regression | The baseline model for binary classification |
| 9 | Algorithm Selection Framework | A repeatable five-step process for choosing the right model |
| 10 | Decision Trees | A model that splits data with yes/no questions to reach a prediction |
| 11 | Random Forest | An ensemble of many decision trees that votes for a more stable answer |
| 12 | Key Tools | scikit-learn, pandas, NumPy, matplotlib, seaborn, Git |

---

## 1. The ML Lifecycle

**What it is:**
A series of stages that every machine learning project moves through, from understanding the business problem all the way to deploying a model and monitoring it in production. It is a cycle, not a straight line -- you loop back to earlier stages as you learn.

**The seven stages:**

| Stage | What Happens |
|-------|-------------|
| **Problem Framing** | Define the question, success metrics, and constraints (latency, fairness, cost) before touching data |
| **Data Ingestion** | Gather data from databases, files, APIs, or streams; document schemas, access, and refresh cadence |
| **EDA and Cleaning** | Explore distributions, spot missing values, duplicates, outliers using pandas, matplotlib, seaborn |
| **Feature Engineering** | Scale, encode, and create new features; this is where domain knowledge lives |
| **Model Training** | Fit a model on training data; randomness enters via shuffles, weight initialization, subsampling |
| **Evaluation** | Measure on held-out data using metrics aligned to the business problem |
| **Deployment and Monitoring** | Serve predictions, log inputs/outputs, watch for data drift, loop back when performance decays |

**Why it matters:**
Interviewers ask this to see if you understand that ML is not just "pick a model and call .fit()." Knowing the lifecycle shows you can scope a project, avoid pitfalls like data leakage, and maintain a model after launch.

**Where it shows up:**
Every industry -- healthcare (predict readmission), finance (fraud detection), e-commerce (recommendation engines), HR analytics (attrition prediction). The stages are universal; only the data and constraints change.

> **Interview Tip:** When asked "walk me through an ML project," map your answer to these stages. It shows structured thinking even if the project was simple.

---

## 2. Reproducibility

**What it is:**
The ability to rerun an experiment with the same inputs and get the same results, whether it is you, a teammate, or an automated pipeline running it.

**Two pillars:**

| Pillar | How |
|--------|-----|
| **Control randomness** | Set explicit random seeds (`random.seed(42)`, `np.random.seed(42)`, `random_state=42` in scikit-learn) |
| **Control versions** | Track code with Git; document data snapshots, library versions, and configuration |

**Why it matters:**
Without reproducibility you cannot debug ("it worked yesterday"), you cannot compare experiments fairly ("is model B actually better, or did the shuffle change?"), and you cannot meet compliance requirements in regulated industries.

**How you use it:**

```python
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
```

**Interesting implementation note:**
The number 42 is convention (a nod to The Hitchhiker's Guide to the Galaxy), not mathematically special. What matters is that you pick a seed, document it, and keep it consistent across experiments. Change the seed deliberately when you want to test stability across multiple trials.

> **Interview Tip:** Mention seed-setting, Git, and environment pinning (requirements.txt, conda env) together. It shows production-level awareness.

---

## 3. Supervised Learning

**What it is:**
A learning paradigm where the training data contains both inputs (features) and known outputs (labels/targets). The model learns the mapping from features to labels and uses it to predict outcomes on new, unseen data.

**Why it matters:**
Most business ML problems are supervised -- "predict this number" or "classify this item." If you have trustworthy labels and a clear prediction goal, supervised learning is the default starting point.

**Two flavors:**

| Type | Target | Example |
|------|--------|---------|
| **Classification** | Discrete categories | Spam or not spam, fraud or legitimate, which digit (0-9) |
| **Regression** | Continuous values | House price, expected revenue, temperature tomorrow |

**How you use it:**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)       # .fit() takes BOTH X and y
predictions = model.predict(X_test)
```

**Where it shows up:**
- **Healthcare:** Predict patient readmission from medical records
- **Finance:** Credit scoring, loan default prediction
- **E-commerce:** Predict customer lifetime value (regression), recommend products (classification)
- **Manufacturing:** Predict equipment failure (binary classification)

---

## 4. Unsupervised Learning

**What it is:**
A learning paradigm where the training data has inputs only -- no labels. The model discovers structure, groupings, or patterns in the data on its own.

**Why it matters:**
Labels are expensive or sometimes impossible to define. Unsupervised methods let you segment customers, reduce data to 2-3 dimensions for visualization, or flag anomalies without needing a labeled "anomaly" column.

**Common tasks:**

| Task | What It Does | Example Algorithm |
|------|-------------|-------------------|
| **Clustering** | Group similar items together | K-Means, DBSCAN |
| **Dimensionality Reduction** | Compress features while preserving structure | PCA, t-SNE, UMAP |
| **Anomaly Detection** | Flag items that do not fit any group | Isolation Forest |

**How you use it:**

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)                     # .fit() takes ONLY X -- no y
cluster_labels = kmeans.labels_
```

**Interesting implementation note:**
Evaluation is indirect. There is no single "accuracy" score for clustering. You use metrics like silhouette score, or you validate clusters by checking if they align with some known business segmentation. Unsupervised results are evidence, not proof, without domain validation.

> **Interview Tip:** A common mistake is treating messy or missing labels as unsupervised. Noisy labels are still a supervised (or weakly supervised) problem. Unsupervised means there genuinely is no target variable.

---

## 5. Reinforcement Learning (Contrast)

**What it is:**
A paradigm where an agent interacts with an environment, takes actions, and receives rewards (or penalties). The agent learns a policy that maximizes cumulative long-term reward through trial and error.

**Why it matters:**
RL is the third major paradigm alongside supervised and unsupervised. You will rarely implement it in a typical ML engineering role, but interviewers expect you to know the distinction and when it applies.

**Key vocabulary:**
- **Agent** -- the learner/decision-maker
- **Environment** -- what the agent interacts with
- **Action** -- a choice the agent makes at each step
- **Reward** -- feedback signal after an action
- **Policy** -- the agent's strategy for choosing actions

**Where it shows up:**
Game-playing (AlphaGo, Atari), robotics (learning to walk), autonomous driving, dynamic pricing, ad placement optimization.

**How it differs:**
- Supervised = labeled (input, correct output) pairs
- Unsupervised = inputs only, find structure
- RL = interact, observe reward, learn over time -- no static dataset of "correct answers"

---

## 6. Linear Regression

**What it is:**
A supervised model that predicts a continuous target as a weighted sum of input features plus an intercept. It draws the best-fit line (or hyperplane) through the data.

**The model:**

```
prediction = w1*feature1 + w2*feature2 + ... + wn*featureN + bias
```

**Why it matters:**
It is the go-to baseline for any regression problem. It is fast, interpretable ("holding other factors constant, each additional year of experience adds $X to salary"), and stable on small to medium tabular data.

**Key evaluation metrics:**

| Metric | Formula | What It Tells You |
|--------|---------|--------------------|
| **R-squared** | 1 - (SS_res / SS_tot) | Fraction of variance explained; 1 = perfect, 0 = no better than the mean |
| **MAE** | mean(\|y_true - y_pred\|) | Average absolute dollar/unit error |
| **RMSE** | sqrt(mean((y_true - y_pred)^2)) | Like MAE but penalizes large errors more heavily |

**How you use it:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("R2:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
```

**Where it shows up:**
- **Real estate:** predict home sale price from square footage, location, age
- **HR analytics:** predict salary from experience, department, projects
- **Operations:** predict demand or shipping time from historical features

**Interesting implementation note:**
The coefficients on standardized features tell you relative importance. If you scale all features first (StandardScaler), the coefficient magnitudes become directly comparable -- the largest absolute coefficient is the most influential feature.

---

## 7. Logistic Regression

**What it is:**
A supervised model for classification (most commonly binary). Despite the name, it is a classifier, not a regressor. It computes a weighted sum of features, passes the result through the sigmoid function, and outputs a probability between 0 and 1.

**The sigmoid function:**

```
probability = 1 / (1 + e^(-z))
where z = w1*x1 + w2*x2 + ... + wn*xn + bias
```

The sigmoid squashes any real number into the range (0, 1), giving you a probability interpretation.

**Why it matters:**
It is the standard first-try classifier for binary problems. It is fast, interpretable (coefficients indicate feature influence on log-odds), and produces calibrated probabilities, not just class labels.

**How you use it:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

labels = model.predict(X_test)           # Class labels: 0 or 1
probs  = model.predict_proba(X_test)[:, 1]  # P(positive class)
```

**Key evaluation metrics:**
Accuracy, precision, recall, F1, ROC-AUC, log-loss. Choice depends on whether you care more about the labels or the quality of the probability estimates, and on the business cost of different error types. (Detailed in Wednesday and Thursday.)

**Where it shows up:**
- **Finance:** approve or deny a loan application
- **Healthcare:** predict whether a patient is at risk of a condition
- **HR:** predict employee attrition (will they leave or stay?)
- **Marketing:** predict whether a user will click or convert

> **Interview Tip:** Know the difference: Linear Regression answers "How much?" Logistic Regression answers "Which class, and how confident?" They share the linear combination of features but differ in output (unbounded number vs bounded probability).

---

## 8. Algorithm Selection Framework

**What it is:**
A repeatable five-step process for narrowing down which ML algorithm to try, rather than guessing or always defaulting to the fanciest model.

**The five steps:**

| Step | Question | Action |
|------|----------|--------|
| 1. **Problem Type** | Classification, regression, clustering, or dimensionality reduction? | Determines your candidate pool |
| 2. **Data Description** | How many rows? Numeric, categorical, or mixed? Sparse? Linear patterns? | List facts, not opinions |
| 3. **Constraints** | Need interpretability? Low latency? Regulatory compliance? | Constraints trim the search space |
| 4. **Baselines First** | How hard is this problem if I barely try? | Fit a DummyClassifier (most frequent class) and a linear model |
| 5. **Shortlist and Compare** | Which candidates beat the baseline? | Same validation splits, multiple metrics, watch for instability across folds |

**Why it matters:**
It turns model selection from opinion into measurement. Complexity should only be added when baselines plateau and diagnostics confirm underfitting -- not because a model sounds impressive.

**How you use it:**

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

dummy = DummyClassifier(strategy="most_frequent")
lr = LogisticRegression(random_state=42, max_iter=1000)

for name, model in [("Dummy", dummy), ("LogReg", lr)]:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"{name}: F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
```

**When to add complexity:**
Only when baselines have plateaued AND diagnostics suggest underfitting AND the uplift justifies the cost. If the root cause is data quality, leakage, or evaluation mistakes, complexity will amplify those problems, not fix them.

> **Interview Tip:** Reference the scikit-learn "Choosing the Right Estimator" flowchart. It is the visual version of this framework and shows you know the standard tooling.

---

## 10. Decision Trees

**What it is:**
A supervised learning model that makes predictions by learning a series of if/then rules from the data. It recursively splits the dataset on the feature and threshold that best separates the target classes (classification) or reduces prediction error (regression), forming a tree structure from a root node down to leaf nodes that hold the final predictions.

**How a split is chosen:**
At each node, the algorithm evaluates every feature and every possible threshold, picking the one that produces the "purest" child nodes. Purity is measured by:

| Criterion | Used For | Idea |
|-----------|----------|------|
| **Gini Impurity** | Classification (default in scikit-learn) | Probability of misclassifying a randomly chosen sample; 0 = pure node |
| **Entropy / Information Gain** | Classification | Measures disorder; a split that reduces entropy the most wins |
| **MSE reduction** | Regression | Split that most reduces the variance of the target in each child |

**Why it matters:**
Decision trees are the foundation for nearly every top-performing tabular ML algorithm (Random Forest, XGBoost, LightGBM, CatBoost). They are also the most interpretable non-linear model -- you can literally draw the decision path and show it to a stakeholder.

**Key hyperparameters:**

| Parameter | What It Controls | Overfitting Impact |
|-----------|-----------------|-------------------|
| `max_depth` | Maximum levels in the tree | Deeper = more complex = higher variance |
| `min_samples_split` | Minimum samples needed to split a node | Higher = more conservative splits |
| `min_samples_leaf` | Minimum samples in a leaf node | Higher = smoother predictions |
| `max_features` | Number of features considered per split | Lower = more randomness, less overfitting |

**How you use it:**

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
```

**Strengths and weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Highly interpretable -- visualize the tree | Prone to overfitting (high variance) without depth limits |
| Handles numeric and categorical features | Unstable -- small data changes can produce a completely different tree |
| No feature scaling required | Greedy splits are locally optimal, not globally optimal |
| Fast to train and predict | Single trees rarely match ensemble performance |

**Where it shows up:**
- **Healthcare:** Clinical decision support ("if blood pressure > X and age > Y, then high risk")
- **Finance:** Credit approval rules that auditors can inspect
- **Operations:** Root cause analysis on manufacturing defect data
- **Any domain requiring explainability:** Regulatory settings where stakeholders need to trace each prediction to a rule

> **Interview Tip:** A single decision tree with `max_depth=None` is the textbook example of high variance / overfitting. In the course evaluation lecture, it is used as the overfitting counterpart to the high-bias logistic regression model on learning curves. Know that an unrestricted tree memorizes training data (near-perfect train score, poor validation score).

---

## 11. Random Forest

**What it is:**
An ensemble method that trains many independent decision trees on random subsets of the data and random subsets of features, then combines their predictions by majority vote (classification) or averaging (regression). The "forest" corrects the instability of any single tree.

**The two sources of randomness:**

| Technique | What It Does |
|-----------|-------------|
| **Bagging (Bootstrap Aggregating)** | Each tree trains on a random sample drawn with replacement from the training set |
| **Feature subsampling** | At each split, only a random subset of features is considered (typically `sqrt(n_features)` for classification) |

Together, these decorrelate the trees so their errors cancel out rather than compound.

**Why it matters:**
Random Forest is the standard "step up" from linear models for tabular data. It handles nonlinear relationships, interactions between features, and mixed feature types with minimal preprocessing. In the course, it is the model trained via Script Mode on Friday and later compared to XGBoost in Week 2.

**Key hyperparameters:**

| Parameter | What It Controls | Default |
|-----------|-----------------|---------|
| `n_estimators` | Number of trees in the forest | 100 |
| `max_depth` | Maximum depth of each tree | None (fully grown) |
| `max_features` | Features considered per split | `sqrt(n_features)` for classification |
| `min_samples_leaf` | Minimum samples in a leaf | 1 |
| `random_state` | Seed for reproducibility | None |

**How you use it:**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
probabilities = rf.predict_proba(X_test)[:, 1]
```

**Feature importance:** Random Forest provides built-in feature importance scores based on how much each feature contributes to reducing impurity across all trees.

```python
importances = rf.feature_importances_
```

**Strengths and weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Much more stable than a single tree (lower variance) | Less interpretable than a single tree (100 trees are hard to visualize) |
| Resistant to overfitting with enough trees | Slower to train/predict than a single tree or linear model |
| No feature scaling required | Each tree is independent, so it cannot correct systematic errors (unlike boosting) |
| Good out-of-the-box performance | Can struggle with very high-dimensional sparse data |
| Built-in feature importance | Memory-heavy for very large forests |

**Random Forest vs Logistic Regression (decision framework):**

| Factor | Logistic Regression | Random Forest |
|--------|--------------------:|:--------------|
| Relationship type | Linear | Nonlinear, interactions |
| Interpretability | Coefficients | Feature importance (less direct) |
| Preprocessing | Needs scaling, encoding | Minimal |
| Speed | Very fast | Moderate |
| When to pick | Baseline, interpretability, linear signal | Nonlinear signal, mixed features, "next step" after linear plateau |

**Where it shows up:**
- **Finance:** Fraud detection (the course's FraudShield scenario uses it as the primary Script Mode model)
- **Healthcare:** Predicting patient outcomes from mixed clinical features
- **E-commerce:** Customer churn prediction, recommendation ranking
- **Environmental science:** Land cover classification from satellite imagery features
- **Any tabular ML competition:** Random Forest is a reliable strong baseline before trying gradient boosting

**How it connects to the rest of the course:**
In Week 2, the course compares Random Forest (trained via Script Mode) to XGBoost (trained as a SageMaker built-in algorithm). XGBoost typically outperforms Random Forest because it uses **sequential boosting** -- each new tree corrects the errors of the ensemble so far -- while Random Forest trains independent trees in parallel. This is the key distinction between **bagging** (Random Forest) and **boosting** (XGBoost, LightGBM, CatBoost).

> **Interview Tip:** Know the bagging vs boosting distinction cold. "Random Forest trains many independent trees on bootstrap samples and averages them (bagging). XGBoost trains trees sequentially where each new tree focuses on the mistakes of the previous ensemble (boosting). Bagging reduces variance; boosting reduces bias." This comparison comes up constantly.

---

## 12. Key Tools (Quick Reference)

| Tool | Role | Why You Use It |
|------|------|----------------|
| **scikit-learn** | ML library | Estimators, preprocessing, metrics, cross-validation -- the standard for tabular ML |
| **pandas** | Data manipulation | DataFrames for loading, cleaning, and exploring tabular data |
| **NumPy** | Numerical computing | Fast array operations, random number generation, linear algebra |
| **matplotlib** | Plotting | Low-level control over charts: histograms, scatter plots, residual plots |
| **seaborn** | Statistical visualization | Higher-level API on top of matplotlib: heatmaps, boxplots, pair plots |
| **Git** | Version control | Track code changes, branch for experiments, roll back mistakes |

---

# Wednesday -- Applied ML Evaluation

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Train / Validation / Test Splits | Three buckets so you learn, tune, and grade without cheating |
| 2 | Stratified Splitting | Preserve class ratios across every split |
| 3 | K-Fold Cross-Validation | Rotate through multiple splits for a more stable score |
| 4 | Loss Functions: MSE vs Cross-Entropy | The scoreboard training optimizes against |
| 5 | Bias-Variance Tradeoff | The tension between too simple and too complex |
| 6 | Learning Curves | A diagnostic plot that reveals whether you need more data, more capacity, or regularization |

---

## 1. Train / Validation / Test Splits

**What it is:**
Dividing your dataset into three non-overlapping subsets, each with a specific role:

| Split | Purpose | When You Use It |
|-------|---------|-----------------|
| **Training set** | Fit model parameters (weights, coefficients) | During `.fit()` |
| **Validation set** | Compare model choices -- hyperparameters, architecture, features | During tuning; can be used many times |
| **Test set** | Final, unbiased estimate of generalization performance | Once, at the very end |

**Why it matters:**
If you tune your model against the test set, you are "peeking at the exam." The test score stops being a fair estimate of how the model will perform on truly unseen data. This is a form of **data leakage**.

**How you use it:**

```python
from sklearn.model_selection import train_test_split

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Second split: of the 80%, take 25% as validation (= 20% of original)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
```

**Analogy:**
Training = experimenting in the kitchen. Validation = taste-testing with friends. Test = the dinner party -- you only serve it once.

> **Interview Tip:** Always mention that you split BEFORE scaling/encoding. Fitting a scaler on the full dataset and then splitting leaks test statistics into training. `fit_transform` on train, `transform` only on val/test.

---

## 2. Stratified Splitting

**What it is:**
A splitting strategy that preserves the class distribution of the target variable in every subset. If your full dataset is 95% legitimate and 5% fraud, each split will also be approximately 95/5.

**Why it matters:**
With imbalanced data, a naive random split can accidentally put most or all of the minority class into one bucket. The training set might have zero fraud examples, or the test set might have none, making evaluation meaningless.

**How you use it:**

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

The `stratify=y` argument does all the work. For cross-validation, use `StratifiedKFold`.

**Where it shows up:**
Any problem where classes are imbalanced -- fraud detection, medical diagnosis (rare diseases), churn prediction, manufacturing defect detection.

---

## 3. K-Fold Cross-Validation

**What it is:**
Split the data into K equal parts (folds). Train on K-1 folds, validate on the remaining fold. Rotate K times so every fold serves as the validation set once. Average the K scores for a more stable performance estimate.

**Why it matters:**
A single train/val split can be "lucky" or "unlucky." Cross-validation gives you a mean and standard deviation, revealing both typical performance and stability. It is essential when data is scarce.

**How you use it:**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
print(f"F1: {scores.mean():.3f} +/- {scores.std():.3f}")
```

**Stratified K-Fold** combines K-fold with stratification -- class ratios are preserved in every fold. This is the default you should use for classification.

**Interesting implementation note:**
Cross-validation is used on the training data only. The held-out test set stays untouched. If instability across folds is high (large standard deviation), that signals a data-size or algorithm-data mismatch problem.

> **Interview Tip:** If asked "how do you evaluate a model," mention cross-validation plus a held-out test set. It shows you understand both stability and honest generalization estimation.

---

## 4. Loss Functions: MSE vs Cross-Entropy

**What it is:**
A loss function is the single number that measures "how wrong" the model's predictions are. Training is the process of minimizing this number. Different problems need different loss functions.

| Loss Function | Use Case | Formula |
|---------------|----------|---------|
| **MSE (Mean Squared Error)** | Regression (continuous targets) | `(1/n) * sum((y_true - y_pred)^2)` |
| **Binary Cross-Entropy** | Binary classification (probabilities) | `-(1/n) * sum(y*log(p) + (1-y)*log(1-p))` |
| **Categorical Cross-Entropy** | Multi-class classification | Extension of binary CE to multiple classes |

**Why it matters:**
Choosing the wrong loss function means training optimizes for the wrong thing. MSE for classification gives weak gradient signals when the model is confidently wrong; cross-entropy gives a steep penalty, forcing faster correction.

**The gradient difference (key insight):**
When the true label is 1 and the model predicts probability p = 0.05 (confidently wrong):
- MSE gradient is proportional to `(p - y)` = small push
- Cross-entropy gradient is proportional to `-1/p` = roughly 10x stronger push

This is why cross-entropy trains classifiers faster and more effectively.

**Rule of thumb:**
- Predicting a number? Use MSE (or MAE/Huber for outlier tolerance).
- Predicting a class? Use cross-entropy.

---

## 5. Bias-Variance Tradeoff

**What it is:**
Two competing sources of prediction error:

| Source | Cause | Symptom |
|--------|-------|---------|
| **Bias** | Model is too simple; systematic error from wrong assumptions | Poor performance on BOTH training and validation data (underfitting) |
| **Variance** | Model is too sensitive to the specific training sample | Great on training data, much worse on validation data (overfitting) |

The tradeoff: increasing model complexity lowers bias but raises variance, and vice versa. The goal is the sweet spot where total error is minimized.

**Why it matters:**
This is the most fundamental concept in model evaluation. Every technique you learn in this course -- regularization, dropout, early stopping, cross-validation -- exists to manage this tradeoff.

**Analogies:**
- High bias = writing one generic "template answer" for every exam question (consistently wrong)
- High variance = memorizing practice problems verbatim (perfect on those, useless on anything new)

**Diagnostic checklist:**

| Observation | Diagnosis | Fix |
|-------------|-----------|-----|
| Train low, val low, both similar | Underfitting (high bias) | More capacity, better features, less regularization |
| Train high, val much lower | Overfitting (high variance) | More data, regularization, simpler model, dropout |
| Train high, val close to train | Good generalization | Ship it (but keep monitoring) |

---

## 6. Learning Curves

**What it is:**
A plot of model performance (e.g., F1, accuracy, or loss) on the y-axis versus training set size (or number of epochs) on the x-axis, showing both training and validation curves.

**Why it matters:**
Learning curves are the primary diagnostic tool for the bias-variance tradeoff. They tell you whether you need more data, more model capacity, or more regularization.

**How to read them:**

| Pattern | Meaning |
|---------|---------|
| Both curves low and converging | **Underfitting** -- the model cannot capture the signal; add features or capacity |
| Training curve high, validation curve much lower, gap stays wide | **Overfitting** -- the model memorizes; regularize or get more data |
| Both curves high, small gap, converging | **Healthy** -- good generalization |
| Validation improves and gap shrinks as training size grows | More data would help |

**How you use it:**

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="f1"
)
# Plot mean +/- std for train_scores and val_scores vs train_sizes
```

> **Interview Tip:** If someone asks "your model has high training accuracy but low test accuracy, what do you do?" -- that is a textbook overfitting question. Mention learning curves, regularization, more data, and simpler architecture.

---

# Thursday -- Neural Networks and Advanced Evaluation

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Perceptrons | The single artificial neuron that started deep learning |
| 2 | Activation Functions | The nonlinearity that lets neural networks learn curves, not just lines |
| 3 | Vanishing Gradient Problem | Why deep sigmoid/tanh networks stop learning |
| 4 | MLPs: Width vs Depth | How many neurons per layer vs how many layers |
| 5 | Backpropagation and Gradient Descent | How neural networks actually learn: forward, measure, backward, update |
| 6 | Learning Rate | The step size that controls how fast (or recklessly) the model updates |
| 7 | Regularization: L1, L2, Dropout | Three ways to fight overfitting in models and networks |
| 8 | Optimizers: SGD vs Adam | The strategy for navigating the loss landscape |
| 9 | Learning Rate Schedulers | Adjusting the step size during training for better convergence |
| 10 | Early Stopping | Halt training when validation stops improving |
| 11 | Precision, Recall, F1 | The metrics that matter when classes are imbalanced |
| 12 | Confusion Matrix | A table that shows exactly which mistakes the model makes |
| 13 | AUC-ROC and Precision-Recall Curves | Threshold-independent evaluation of classification models |
| 14 | SHAP Values | Explain why the model made a specific prediction |

---

## 1. Perceptrons

**What it is:**
The simplest artificial neuron. It takes multiple inputs, multiplies each by a learned weight, sums them up, adds a bias, and passes the result through an activation function to produce an output.

**The math:**

```
z = w1*x1 + w2*x2 + ... + wn*xn + b      (weighted sum + bias)
output = activation(z)
```

**Biological analogy:**
- Dendrites = inputs
- Cell body = weighted sum
- Firing threshold = activation function
- Axon = output

The original Rosenblatt perceptron used a step function (output 1 if z >= 0, else 0), making it a binary linear classifier. Modern networks replace the step with smooth, differentiable activations.

**Why it matters:**
Every neuron in every deep learning model is a perceptron with a different activation. Understanding this building block is essential for understanding everything built on top of it.

---

## 2. Activation Functions

**What it is:**
A function applied after the weighted sum in each neuron that introduces nonlinearity. Without activations, stacking layers of linear transforms is still just one big linear transform -- the network could never learn curved decision boundaries.

**The lineup:**

| Function | Formula | Output Range | Default Use |
|----------|---------|-------------|-------------|
| **ReLU** | max(0, z) | [0, infinity) | Hidden layers -- the default starting point |
| **Leaky ReLU** | z if z > 0, else 0.01*z | (-infinity, infinity) | Hidden layers when dying ReLU is a concern |
| **GELU** | z * P(Z <= z) (smooth) | (-small, infinity) | Hidden layers in transformers (BERT, GPT) |
| **Sigmoid** | 1 / (1 + e^(-z)) | (0, 1) | Binary output layer (probability) |
| **Tanh** | (e^z - e^(-z)) / (e^z + e^(-z)) | (-1, 1) | When zero-centered output matters (legacy hidden layers) |
| **Softmax** | e^(zi) / sum(e^(zj)) | (0, 1), sums to 1 | Multi-class output layer (probability distribution) |

**Why it matters:**
Choosing the right activation is a practical design decision. ReLU is fast and avoids vanishing gradients for hidden layers. Sigmoid and softmax belong at the output layer for classification. Using sigmoid in deep hidden layers leads to the vanishing gradient problem.

> **Interview Tip:** If asked "why ReLU?" -- its derivative is 1 for positive inputs and 0 for negative inputs. This constant gradient of 1 prevents the vanishing gradient problem in deep networks. The tradeoff is "dying ReLU": neurons that get stuck at 0 and never recover.

---

## 3. Vanishing Gradient Problem

**What it is:**
When activations like sigmoid or tanh saturate (output near 0 or 1 for sigmoid, near -1 or 1 for tanh), their derivatives become extremely small. During backpropagation, gradients are multiplied layer by layer. Many tiny numbers multiplied together shrink to near zero, so early layers receive almost no learning signal.

**Why it matters:**
It explains why deep networks historically could not be trained effectively and why ReLU was a breakthrough. It is also why modern architectures use skip connections (ResNets) and careful initialization.

**The fix:**
- Use **ReLU** (derivative = 1 for positive inputs)
- Use **Leaky ReLU** or **GELU** to avoid dead neurons
- Use **batch normalization** or **skip connections** in very deep networks
- Proper weight initialization (He initialization for ReLU, Xavier for tanh)

---

## 4. MLPs: Width vs Depth

**What it is:**
A Multi-Layer Perceptron (MLP) is a fully connected feedforward neural network with one or more hidden layers. The two architectural knobs are:

| Dimension | Meaning | Effect |
|-----------|---------|--------|
| **Width** | Neurons per hidden layer | More features processed simultaneously per step |
| **Depth** | Number of hidden layers | Hierarchical, compositional representations (edges, then shapes, then objects) |

**The Universal Approximation Theorem:**
A single hidden layer with enough neurons can approximate any continuous function. However, depth is often exponentially more parameter-efficient -- a deep network can represent the same complexity with far fewer total parameters than one massive wide layer.

**Parameter counting (fully connected layer):**

```
parameters = (inputs * outputs) + outputs
           = (m * n) + n    for a layer going from m to n neurons
```

Example: Input 784 -> Hidden 128 -> Hidden 128 -> Output 10
- Layer 1: 784 * 128 + 128 = 100,480
- Layer 2: 128 * 128 + 128 = 16,512
- Layer 3: 128 * 10 + 10   = 1,290
- Total: 118,282

**Starting heuristics:**
- Tabular data: 1-4 hidden layers, 64-512 neurons each
- Start simple (1-2 layers), add depth/width only if training loss shows underfitting
- If validation loss diverges from training loss, reduce capacity or add regularization

---

## 5. Backpropagation and Gradient Descent

**What it is:**
The two-part mechanism that trains neural networks:

1. **Gradient descent** is the optimization strategy: update each weight in the direction that reduces the loss, scaled by a learning rate
2. **Backpropagation** is the algorithm that efficiently computes the gradient of the loss with respect to every weight using the chain rule, working backward from the output to the input

**The training loop (five steps, repeated every batch):**

```
1. Forward pass    -- compute predictions
2. Compute loss    -- scalar measure of "how wrong"
3. Backpropagation -- compute gradients via chain rule (backward pass)
4. Update weights  -- w_new = w_old - learning_rate * gradient
5. Clear gradients -- frameworks accumulate by default; reset before next batch
```

**Why it matters:**
This is how every neural network learns. The chain rule propagates error information from the output back through every layer, and gradient descent uses that information to adjust weights. Understanding this loop is essential for debugging training issues.

**The weight update rule:**

```
w_new = w_old - lr * (dL/dw)
```

Where `dL/dw` is the partial derivative of the loss with respect to that weight.

**Debugging with the training loop:**

| Symptom | Likely Cause |
|---------|-------------|
| Loss is flat | Learning rate too small, capacity too low, or vanishing gradients |
| Loss explodes to infinity | Learning rate too large or exploding gradients |
| Loss oscillates wildly | Learning rate too large or batch size too small |

> **Interview Tip:** Autograd (automatic differentiation) is how frameworks like PyTorch and TensorFlow implement backpropagation. They build a computational graph during the forward pass and traverse it in reverse to compute gradients. You do not write the chain rule by hand.

---

## 6. Learning Rate

**What it is:**
The scalar that controls how large each weight update step is. Too high and the model overshoots the minimum (loss oscillates or explodes). Too low and training is painfully slow or gets stuck in a poor local minimum.

**Typical starting values:**

| Optimizer | Common Default LR |
|-----------|-------------------|
| SGD | 0.01 -- 0.1 |
| Adam | 0.001 (1e-3) |

The learning rate is often the single most important hyperparameter to tune.

---

## 7. Regularization: L1, L2, Dropout

**What it is:**
Techniques that penalize model complexity to reduce overfitting. They constrain the model so it learns the signal rather than memorizing noise.

### L2 Regularization (Ridge)

**Penalty:** Add `lambda * sum(w_i^2)` to the loss.

**Effect:** Shrinks all coefficients toward zero but rarely makes them exactly zero. Like "turning down the volume" on every feature proportionally.

**When to use:** General-purpose regularization for linear models and neural networks. Good when you have many correlated features that are all somewhat useful.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)    # alpha = lambda (regularization strength)
```

### L1 Regularization (Lasso)

**Penalty:** Add `lambda * sum(|w_i|)` to the loss.

**Effect:** Drives some coefficients to exactly zero, effectively performing feature selection. Some features get "muted" entirely.

**When to use:** When you suspect many features are irrelevant and you want the model to pick the important ones automatically.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01, max_iter=5000)
# Check model.coef_ -- many will be exactly 0.0
```

### Elastic Net

**What it is:** A combination of L1 and L2 penalties. Useful when features are correlated and pure L1 is unstable.

### Dropout (Neural Networks)

**What it is:** During each training step, randomly set a fraction of neurons to zero. At inference time, use all neurons but scale outputs by `(1 - drop_rate)`.

**Effect:** Forces the network to learn redundant representations (no single neuron can be relied upon). Acts like training an ensemble of subnetworks.

**Typical rate:** 0.2 -- 0.5 for hidden layers. Applied during training only.

**In scikit-learn (C parameter):**
scikit-learn uses `C = 1 / lambda`. **Smaller C = stronger regularization.** This is the inverse convention.

```python
LogisticRegression(penalty="l2", C=0.1)   # Strong L2
LogisticRegression(penalty="l1", C=0.1, solver="saga")  # Strong L1
```

> **Interview Tip:** Know all three and when to reach for each. L2 = stable shrinkage. L1 = sparse feature selection. Dropout = neural network ensemble trick. Early stopping (below) is also a form of regularization.

---

## 8. Optimizers: SGD vs Adam

**What it is:**
The algorithm that determines how weights are updated at each step. Different optimizers handle the loss landscape differently.

### SGD (Stochastic Gradient Descent)

Compute the gradient on a mini-batch (not the full dataset) and step in the opposite direction.

```
w = w - lr * gradient
```

**Stochastic noise** from mini-batches can actually help escape sharp minima and find flatter regions that generalize better.

### SGD + Momentum

Accumulate a velocity from past gradients. Accelerate in consistent directions, dampen oscillation.

```
velocity = momentum * velocity + gradient       (momentum ~ 0.9)
w = w - lr * velocity
```

### Adam (Adaptive Moment Estimation)

Maintains per-parameter adaptive learning rates using first and second moment estimates of the gradient.

```
m = beta1 * m + (1 - beta1) * gradient          (first moment, beta1 ~ 0.9)
v = beta2 * v + (1 - beta2) * gradient^2         (second moment, beta2 ~ 0.999)
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)     (after bias correction)
```

**When to use each:**

| | SGD + Momentum | Adam |
|--|---------------|------|
| **Strengths** | Often better final accuracy with tuning; lower memory; proven for vision | Works well out of the box; handles sparse/noisy gradients; minimal tuning |
| **Weaknesses** | Requires careful LR schedule tuning | Higher memory (2 extra tensors per weight); sometimes generalizes slightly worse |
| **Best for** | Training vision models from scratch when you can invest in tuning | NLP, transformers, prototyping, fine-tuning, any time you want fast iteration |
| **Default LR** | 0.01 -- 0.1 | 0.001 |

> **Interview Tip:** "Adam for fast iteration, SGD+momentum for maximum performance when you have time to tune." This is the standard industry rule of thumb.

---

## 9. Learning Rate Schedulers

**What it is:**
Strategies that adjust the learning rate during training rather than keeping it fixed. Starting high for fast progress and reducing it later for fine-grained convergence.

| Scheduler | How It Works | Best For |
|-----------|-------------|----------|
| **Step Decay** | Multiply LR by a factor (e.g., 0.5) every N epochs | Predictable training length |
| **Cosine Annealing** | Smoothly decrease LR following a cosine curve to near zero | Fixed-epoch budgets; popular in vision |
| **Reduce on Plateau** | Cut LR when a monitored metric (val loss) stops improving for N epochs | Adaptive; good general-purpose choice |

---

## 10. Early Stopping

**What it is:**
A regularization technique that monitors a validation metric (usually validation loss) during training and stops when the metric has not improved for a set number of consecutive checks (the "patience"). The model weights from the best-performing epoch are restored.

**Key parameters:**

| Parameter | Role |
|-----------|------|
| **Patience** | How many epochs without improvement before stopping |
| **min_delta** | Minimum change to count as an improvement |
| **Restore best weights** | Roll back to the checkpoint with the lowest validation loss, not the last epoch |

**Why it matters:**
Training too long means the model starts memorizing the training data (validation loss goes up while training loss continues to go down). Early stopping automatically finds the point where the model generalizes best.

**The pattern:**

```
For each epoch:
    Train on training set
    Evaluate on validation set
    If val_loss improved by at least min_delta:
        Save current weights as best
        Reset patience counter
    Else:
        Increment patience counter
    If patience counter >= patience:
        Stop training
        Restore best weights
```

> **Interview Tip:** Early stopping is regularization by controlling training time. Mention it alongside L2/dropout as part of your overfitting prevention toolkit.

---

## 11. Precision, Recall, F1 Score

**What it is:**
Classification metrics that go beyond accuracy to measure the quality of predictions for each class, especially important when classes are imbalanced.

**The formulas:**

```
Precision = TP / (TP + FP)
  "Of everything the model flagged as positive, what fraction was actually positive?"

Recall    = TP / (TP + FN)
  "Of everything that was actually positive, what fraction did the model catch?"

F1 Score  = 2 * Precision * Recall / (Precision + Recall)
  "Harmonic mean of precision and recall -- penalizes lopsided pairs"
```

**Why accuracy fails under imbalance:**
If 95% of transactions are legitimate, a model that always predicts "legitimate" gets 95% accuracy but catches zero fraud. Accuracy looks great; the model is useless.

**Which metric to emphasize:**

| Scenario | Priority | Reason |
|----------|----------|--------|
| Fraud detection | **Recall** | Missing fraud (FN) is more costly than false alarms (FP) |
| Spam filter | **Precision** | Flagging legitimate email as spam (FP) angers users |
| Balanced trade-off | **F1** | When both types of errors matter roughly equally |

**How you use it:**

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print(classification_report(y_test, y_pred))
```

> **Interview Tip:** When asked about metrics, always tie your choice to business cost. "In this fraud scenario, a false negative costs the bank $X, so we optimize for recall with an acceptable precision floor."

---

## 12. Confusion Matrix

**What it is:**
A 2x2 table (for binary classification) that shows exactly how many predictions fell into each category:

```
                    Predicted Negative    Predicted Positive
Actual Negative         TN                    FP
Actual Positive         FN                    TP
```

| Cell | Name | Meaning |
|------|------|---------|
| **TN** | True Negative | Correctly predicted negative |
| **FP** | False Positive | Incorrectly predicted positive (Type I error) |
| **FN** | False Negative | Incorrectly predicted negative (Type II error) |
| **TP** | True Positive | Correctly predicted positive |

**Why it matters:**
It is the most detailed view of classification performance at a single threshold. Every metric (precision, recall, F1, accuracy) is derived from these four numbers.

**How you use it:**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

**Where it shows up:**
Every classification problem. In interviews, you may be asked to compute precision and recall by hand from a confusion matrix.

---

## 13. AUC-ROC and Precision-Recall Curves

**What it is:**
Two threshold-independent ways to evaluate a classifier that outputs probabilities, not just labels.

### ROC Curve and AUC

- **ROC curve:** Plots True Positive Rate (Recall) vs False Positive Rate as you sweep the classification threshold from 0 to 1
- **AUC (Area Under the ROC Curve):** Single number summary; 1.0 = perfect separation, 0.5 = random guessing

```
FPR = FP / (FP + TN)       "Of actual negatives, how many did we falsely flag?"
TPR = TP / (TP + FN)       "Of actual positives, how many did we catch?" (= Recall)
```

**Interpretation of AUC:** The probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example.

### Precision-Recall Curve

- Plots precision vs recall as threshold varies
- **Average Precision (AP):** Area under the PR curve
- Often more informative than ROC under heavy class imbalance because the large number of true negatives can make ROC look optimistically good

**When to use which:**

| Metric | Best When |
|--------|-----------|
| AUC-ROC | Balanced classes or comparing ranking quality across models |
| PR Curve / AP | Imbalanced classes where you care about the minority class |

**How you use it:**

```python
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
```

> **Interview Tip:** AUC tells you about ranking quality across all thresholds. But in production you deploy at one specific threshold. After picking your threshold, report precision, recall, F1, and the confusion matrix at that operating point.

---

## 14. SHAP Values

**What it is:**
SHAP (SHapley Additive exPlanations) is a method from cooperative game theory that assigns each feature a contribution value for a specific prediction. The SHAP values for all features, plus a baseline (average model output), sum exactly to the model's prediction for that instance.

**Why it matters:**
- **Debugging:** Verify the model uses sensible features, not data leakage or proxies
- **Fairness:** Detect if protected attributes (race, gender) are driving predictions
- **Regulatory:** Industries like finance and healthcare increasingly require explainability
- **Trust:** Stakeholders are more likely to adopt a model they can inspect

**Two levels of explanation:**

| Level | What It Shows | Plot Type |
|-------|--------------|-----------|
| **Global** | Which features are most important overall | Bar plot of mean absolute SHAP values |
| **Local** | Why this specific prediction was made | Waterfall or force plot for one instance |

**How you use it:**

```python
import shap

explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled)   # Global: beeswarm
shap.plots.bar(shap_values)                      # Global: mean |SHAP|
```

**The Shapley intuition:**
Think of features as "players" on a team and the prediction as the "payout." SHAP fairly distributes credit by considering every possible combination of features and their marginal contributions. A positive SHAP value pushes the prediction toward the positive class; negative pushes it toward the negative class.

> **Interview Tip:** SHAP is the most asked-about explainability method. Know that it is model-agnostic (or model-specific via TreeExplainer, LinearExplainer), that values are additive and sum to the prediction, and that correlation does not equal causation -- a high SHAP value for a feature does not mean that feature causes the outcome.

---

# Friday -- SageMaker Foundations, Training, and Deep Learning

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | SageMaker Ecosystem | AWS's managed platform that ties the ML lifecycle to cloud services |
| 2 | Studio Domains and User Profiles | The organizational unit that governs network, storage, and roles for a team |
| 3 | Studio vs Classic vs Canvas | Three interfaces for three personas: code-heavy, legacy, no-code |
| 4 | IAM and Least-Privilege | The "badge system" that controls what SageMaker can access |
| 5 | The SageMaker ML Lifecycle | Five stages: Prepare, Build, Train/Tune, Deploy, Monitor |
| 6 | JumpStart Pre-built Models | One-click deploy of pre-trained or fine-tunable models |
| 7 | BYOM vs BYOS Spectrum | Choosing between built-in algorithms, Script Mode, and custom containers |
| 8 | Script Mode Structure | How your Python script talks to the SageMaker training container |
| 9 | Estimators and Configurations | The SDK object that wraps a training job's entire configuration |
| 10 | Training Job Anatomy | The seven steps SageMaker executes when you call .fit() |
| 11 | Model Artifacts and S3 | What gets saved after training and where it lives |
| 12 | CNNs for Image Data | Convolutional neural networks that preserve spatial structure |
| 13 | Encoder-Decoder Architectures | Sequence-to-sequence models that map variable-length inputs to variable-length outputs |

---

## 1. SageMaker Ecosystem and Core Services

**What it is:**
Amazon SageMaker is a fully managed AWS service that provides tools for every stage of the ML lifecycle -- from data preparation through training, deployment, and monitoring -- without you having to manage the underlying infrastructure.

**Core services map:**

| ML Stage | SageMaker Service | AWS Companion |
|----------|------------------|---------------|
| Data storage | -- | **Amazon S3** (data + model artifacts) |
| EDA / experimentation | **SageMaker Studio** (JupyterLab IDE) | -- |
| Feature engineering | **Processing Jobs** | -- |
| Data labeling | **Ground Truth** | -- |
| Training | **Training Jobs** (managed compute) | **CloudWatch** (logs/metrics) |
| Hyperparameter tuning | **HPO Jobs** | -- |
| Deployment | **Endpoints** (real-time HTTPS) | -- |
| Monitoring | **Model Monitor** | **CloudWatch** (dashboards/alerts) |
| Governance | **Model Registry** | **IAM** (roles/policies) |

**Automation:** **SageMaker Python SDK** (`import sagemaker`) and **AWS CLI** (`aws sagemaker ...`) let you script everything the console does.

**Cost awareness:**
SageMaker has a Free Tier with limits. Common instance types: `ml.t3.medium` (notebooks), `ml.m5.xlarge` (training/inference). Always check **Billing and Cost Management** and clean up endpoints and resources when done.

---

## 2. Studio Domains and User Profiles

**What it is:**
A **Domain** is the top-level organizational unit in SageMaker. Think of it as a shared "floor plan" that defines the network configuration (VPC/subnets), default execution role, and shared storage (Amazon EFS) for a team. A **User Profile** is a personal workspace within that domain -- each user gets their own EFS home directory and can optionally override the domain's default role.

**Key components:**

| Component | What It Provides |
|-----------|-----------------|
| **VPC / Subnets** | Network isolation and security |
| **Default Execution Role** | IAM role that SageMaker assumes for all operations by default |
| **Amazon EFS** | Persistent file storage (survives kernel restarts, unlike ephemeral compute) |
| **User Profile** | Per-user home directory, optional role override, personal settings |

**Console flow:**
SageMaker > Domains > Create Domain > **Quick Setup** (default VPC, auto-created role, one profile) or **Standard Setup** (custom VPC, encryption, custom roles).

**Why it matters:**
Domains enforce governance at the infrastructure level. Everyone on the team shares the same network boundary and default permissions, while each user gets their own isolated workspace. This is how organizations manage reproducibility and security at scale.

---

## 3. Studio vs Studio Classic vs Canvas

| Interface | Target User | Code Required? | Key Capabilities |
|-----------|-------------|---------------|-----------------|
| **SageMaker Studio** | Data scientists, ML engineers | Yes | JupyterLab, experiments, training, registry, pipelines |
| **Studio Classic** | Legacy users | Yes | Older architecture (separate app instances); being phased out |
| **SageMaker Canvas** | Business analysts | No | Point-and-click ML; same Domain as Studio |

**Why it matters:**
In an interview, knowing that Canvas exists shows you understand the full spectrum of users. Not everyone who needs ML predictions can write Python.

---

## 4. IAM and Least-Privilege Practices

**What it is:**
AWS Identity and Access Management (IAM) is the "badge system" for AWS. In SageMaker, the most important IAM concept is the **execution role** -- the IAM role that SageMaker assumes when it runs training jobs, deploys models, or accesses S3 data. SageMaker does not use your personal AWS credentials.

**Key IAM building blocks:**

| Concept | What It Is |
|---------|-----------|
| **User** | A human identity with credentials |
| **Role** | An identity that a service (like SageMaker) assumes; has temporary credentials |
| **Policy** | A JSON document specifying allow/deny rules for specific actions on specific resources |
| **Trust Policy** | Defines which principals (e.g., `sagemaker.amazonaws.com`) can assume the role |

**Execution role anatomy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

**Least-privilege best practices:**

| Practice | Why |
|----------|-----|
| Start with narrow permissions, widen only as needed | Limits blast radius of misconfiguration |
| One role per workload type (training role, inference role) | Prevents training jobs from accessing production endpoints |
| Avoid `"Resource": "*"` | Grants access to everything in the account |
| Use **CloudTrail** to audit denied actions | Find and fix permission gaps instead of over-granting |
| Use tag-based access control for multi-project accounts | e.g., condition on `sagemaker:ResourceTag/project` |

**Common starter mistake:**
`AmazonSageMakerFullAccess` is convenient for learning but far too broad for production. A custom policy should scope actions (e.g., `s3:GetObject`, `s3:PutObject`) to specific bucket ARNs.

> **Interview Tip:** If asked about security in ML pipelines, mention execution roles, least-privilege policies, VPC isolation, and encryption (KMS). This shows you think about production, not just notebooks.

---

## 5. The SageMaker ML Lifecycle

**What it is:**
SageMaker organizes its services into five stages that map to the general ML lifecycle:

```
Prepare --> Build --> Train & Tune --> Deploy --> Monitor
   ^                                                |
   |________________________________________________|
                    (iterate)
```

| Stage | What You Do | SageMaker Services |
|-------|------------|-------------------|
| **Prepare** | Upload data to S3, run processing/transform jobs, label data | S3, Processing Jobs, Ground Truth |
| **Build** | Write/explore in notebooks, browse pre-trained models | Studio, JumpStart |
| **Train & Tune** | Run managed training jobs, hyperparameter tuning | Training Jobs, HPO |
| **Deploy** | Create real-time endpoints, batch transform, serverless inference | Models, Endpoint Configs, Endpoints |
| **Monitor** | Track data drift, model quality degradation, trigger retraining | Model Monitor |

**Why it matters:**
This five-stage framework is how AWS structures its SageMaker documentation, console UI, and certification exams. Knowing where each service fits shows you understand the platform holistically.

---

## 6. JumpStart Pre-built Models

**What it is:**
A curated catalog inside SageMaker Studio of pre-trained and fine-tunable models, plus solution templates with notebooks and deployment configs. Models come from AWS, Hugging Face, Meta, Stability AI, and others.

**Categories:** Text generation, image classification, object detection, sentence embeddings, tabular models, and more.

**Deploy flow (console):**
1. Open Studio > JumpStart
2. Browse or search for a model
3. Review endpoint name, instance type, execution role
4. Click **Deploy** > wait for status **InService**
5. Verify under **Inference > Endpoints**
6. Use the companion notebook to call `predict()`

**Cleanup (mandatory):**
Delete the **Endpoint** first, then the **Model**. Forgetting to delete endpoints is the number-one cause of unexpected SageMaker charges.

**When to use JumpStart vs custom training:**

| Scenario | Path |
|----------|------|
| Quick proof-of-concept | Pre-trained model, deploy as-is |
| Need domain-specific accuracy | Fine-tune a JumpStart model on your data |
| Unique problem or learning the pipeline | Custom training from scratch |

---

## 7. BYOM vs BYOS (Script Mode) Spectrum

**What it is:**
SageMaker offers three levels of customization for training:

| Level | You Provide | SageMaker Provides |
|-------|------------|-------------------|
| **Built-in Algorithms** | Data + hyperparameters | The entire algorithm and container |
| **Script Mode (BYOS)** | Your `.py` training script | A managed framework container (scikit-learn, PyTorch, TensorFlow, XGBoost) |
| **BYOM (Bring Your Own Model)** | A full Docker image pushed to ECR | Only the compute infrastructure |

**Script Mode is the recommended middle ground.** You write standard Python code, SageMaker handles the container, data transfer, and infrastructure.

**BYOM** is for when you need custom system libraries, non-standard frameworks, or full control over the container. You build a Docker image, push it to **Amazon ECR**, and SageMaker runs it according to a specific container contract.

> **Interview Tip:** Know the spectrum. Most teams use Script Mode. BYOM is for edge cases. Built-in algorithms are fastest to start but least flexible.

---

## 8. Script Mode Structure

**What it is:**
The contract between your Python script and the SageMaker training container. SageMaker maps your data from S3 to specific paths inside the container and expects your script to save model artifacts to a specific output directory.

**Container layout (`/opt/ml/`):**

```
/opt/ml/
  input/
    config/
      hyperparameters.json    <-- your hyperparams as JSON
      resourceconfig.json     <-- instance info
    data/
      train/                  <-- training channel data from S3
      validation/             <-- validation channel data from S3
  model/                      <-- SAVE YOUR MODEL HERE
  output/
    failure                   <-- write error message here if script fails
```

**Script requirements:**

| Requirement | Why |
|-------------|-----|
| `if __name__ == "__main__":` guard | SageMaker invokes the script directly as `python train.py` |
| `argparse` for hyperparameters | SageMaker passes hyperparameters as CLI arguments (`--n-estimators 100`) |
| Read data from `SM_CHANNEL_TRAIN` env var | Points to `/opt/ml/input/data/train/` |
| Save model to `SM_MODEL_DIR` env var | Points to `/opt/ml/model/`; contents become `model.tar.gz` |

**Minimal Script Mode example:**

```python
import argparse, os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "data")
    model_dir = os.environ.get("SM_MODEL_DIR", "model")

    df = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
```

> **Interview Tip:** Know the four env vars: `SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION`, `SM_MODEL_DIR`, `SM_OUTPUT_DATA_DIR`. This shows you have actually written SageMaker training code, not just used the console.

---

## 9. Estimators and Configurations

**What it is:**
The SageMaker Python SDK's **Estimator** object wraps a training job's entire configuration -- the script, container image, instance type, hyperparameters, input channels, and output path -- into a single Python object. Calling `.fit()` on it launches a managed cloud training job.

**Key configuration fields:**

| SDK Parameter | Console Equivalent | Example |
|--------------|-------------------|---------|
| `entry_point` | Entry point script | `"train.py"` |
| `source_dir` | Source directory | `"code/"` |
| `role` | IAM Role ARN | `"arn:aws:iam::123:role/SageMakerRole"` |
| `instance_type` | Instance type | `"ml.m5.xlarge"` |
| `instance_count` | Instance count | `1` |
| `volume_size` | Volume size (GB) | `30` |
| `max_run` | Max runtime (seconds) | `3600` |
| `hyperparameters` | Hyperparameters | `{"n-estimators": 100}` |
| `output_path` | S3 output path | `"s3://bucket/output"` |

**Calling .fit():**

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point="train.py",
    source_dir="code/",
    role=role,
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
    hyperparameters={"n-estimators": 100}
)

estimator.fit({"train": "s3://bucket/data/train/"})
```

**Framework estimators:** `SKLearn`, `PyTorch`, `TensorFlow`, `XGBoost`. For BYOM, use the generic `Estimator` with `image_uri`.

**Local mode:** Set `instance_type="local"` and data to `file://./data/` for fast iteration before spending money on cloud instances.

---

## 10. Training Job Anatomy

**What it is:**
The seven steps SageMaker executes when you call `estimator.fit()`:

| Step | Status | What Happens |
|------|--------|-------------|
| 1 | **Starting** | Provision EC2 instance(s) with requested type and volume |
| 2 | **Downloading** | Pull container image from Amazon ECR |
| 3 | **Downloading** | Download input data from S3 to `/opt/ml/input/data/<channel>/` |
| 4 | **Training** | Inject and run your script with env vars and CLI hyperparameters |
| 5 | **Uploading** | Package `/opt/ml/model/` into `model.tar.gz` and upload to S3 |
| 6 | **Uploading** | Stream logs and metrics to CloudWatch |
| 7 | **Completed** | Tear down infrastructure; status becomes Completed (or Failed) |

**Console debugging checklist:**

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Access Denied | IAM role lacks S3 or ECR permissions | Update execution role policy |
| FileNotFoundError | Wrong S3 path or channel name mismatch | Verify `.fit()` channel names match script's env vars |
| ModuleNotFoundError | Missing dependency | Add `requirements.txt` to source directory |
| Non-zero exit | Script crashed | Check CloudWatch logs at `/aws/sagemaker/TrainingJobs` |
| Timeout | Exceeded `max_run` | Increase `max_run` or use a larger instance |

**After training:** `estimator.model_data` gives you the S3 URI of the `model.tar.gz` artifact.

> **Interview Tip:** Be able to walk through these seven steps from memory. It shows you understand what happens behind `.fit()` and where to look when things break.

---

## 11. Model Artifacts and S3 Storage

**What it is:**
After training completes, SageMaker tars everything in `/opt/ml/model/` into a file called `model.tar.gz` and uploads it to S3. This archive is the deployable artifact.

**S3 path pattern:**

```
s3://<bucket>/<output_path_prefix>/<training-job-name>/output/model.tar.gz
```

**What goes in the archive:**
Whatever your training script saved to `SM_MODEL_DIR`. For scikit-learn, typically `model.pkl` (via joblib). For PyTorch, model weights + config. For TensorFlow, the full SavedModel directory.

**Important:** The serving container at deployment time expects the exact same file structure and format. If you saved with `joblib.dump`, the inference container must load with `joblib.load`.

**Versioning:** Each training job has a unique name, so each job produces a distinct S3 path. For formal version management, use the **Model Registry** (covered in Week 2).

---

## 12. CNNs for Image Data

**What it is:**
Convolutional Neural Networks are specialized architectures designed for data with spatial structure (primarily images). Instead of connecting every input pixel to every neuron (which explodes in parameter count), CNNs use small learnable filters that slide across the image, detecting local patterns like edges, textures, and shapes.

**Why not a plain MLP for images?**
A 128x128 RGB image has 49,152 input values. A single fully connected layer with 256 neurons would need 49,152 x 256 + 256 = ~12.6 million parameters in just the first layer. CNNs dramatically reduce this through local connectivity and weight sharing.

**Core mechanisms:**

| Mechanism | What It Does |
|-----------|-------------|
| **Convolution** | A small filter (e.g., 3x3) slides across the image, computing dot products to produce a feature map |
| **Weight sharing** | The same filter is reused across all spatial positions -- same weights detect the same pattern anywhere in the image |
| **Pooling** | Downsamples feature maps (typically max pooling 2x2, stride 2) to reduce size and add translation invariance |
| **Activation** | ReLU after each convolution to introduce nonlinearity |

**Output size formula:**

```
W_out = floor((W_in - kernel_size + 2 * padding) / stride) + 1
```

Example: Input 32x32, kernel 3x3, stride 1, padding 0:
`W_out = (32 - 3 + 0) / 1 + 1 = 30`

With padding=1 ("same" padding):
`W_out = (32 - 3 + 2) / 1 + 1 = 32` (preserves spatial dimensions)

**Parameter count for a convolutional layer:**

```
params = C_out * (C_in * kernel_h * kernel_w + 1)
```

Example: 16 filters of size 3x3 on a 3-channel (RGB) input:
`params = 16 * (3 * 3 * 3 + 1) = 16 * 28 = 448` (compare to millions for a dense layer)

**The typical CNN stack:**

```
[Conv -> ReLU -> Pool] x N  -->  Flatten  -->  [FC -> ReLU] x M  -->  Output
```

Early layers learn low-level features (edges, corners). Deeper layers compose those into higher-level features (eyes, wheels, letters).

**Data augmentation:** Apply random transformations (flips, rotations, crops, color jitter) to training images to increase effective dataset size and reduce overfitting. Only augment training data, never validation or test. Augmentations must be semantically valid (do not horizontally flip digits 6 and 9).

**Where it shows up:**
- **Medical imaging:** Classify X-rays, detect tumors in MRI scans
- **Autonomous driving:** Detect pedestrians, signs, lane markings
- **Retail:** Product recognition, visual search
- **Security:** Face recognition, document verification
- **Agriculture:** Crop disease detection from drone imagery

> **Interview Tip:** Know the output size formula, understand weight sharing vs fully connected, and explain why pooling helps (downsamples, adds invariance). Mention that modern CNNs use architectures like ResNet (skip connections) and MobileNet (depthwise separable convolutions for efficiency).

---

## 13. Encoder-Decoder Architectures

**What it is:**
A neural network architecture for sequence-to-sequence (seq2seq) tasks where the input sequence and output sequence can have different lengths. An **encoder** reads the entire input and compresses it into a fixed-size vector (the "context vector" or "thought vector"). A **decoder** then generates the output sequence one token at a time, using the context vector as its starting point.

**The architecture:**

```
Input Sequence --> [Encoder (LSTM/GRU)] --> Context Vector --> [Decoder (LSTM/GRU)] --> Output Sequence
```

**Key vocabulary:**

| Term | Meaning |
|------|---------|
| **Encoder** | Processes the input sequence; its final hidden state becomes the context vector |
| **Decoder** | Generates output tokens one at a time; initialized with the context vector |
| **Context vector** | Fixed-size summary of the entire input; the "bridge" between encoder and decoder |
| **Teacher forcing** | During training, feed the decoder the correct previous token (not its own prediction) for faster, more stable training |
| **Autoregressive** | At inference time, feed the decoder its own previous prediction since there are no ground-truth tokens available |

**The bottleneck problem:**
Compressing an entire input sequence into a single fixed-size vector is lossy, especially for long sequences. Important information from early in the input can be lost. This limitation motivated the development of **attention mechanisms**, which let the decoder look back at all encoder hidden states, not just the final one. Attention led directly to the **Transformer architecture** (covered in Week 2).

**Training vs inference mismatch:**
With teacher forcing, the decoder always sees correct previous tokens during training. At inference, it sees its own (possibly wrong) predictions. **Scheduled sampling** mitigates this by gradually mixing in the model's own predictions during training.

**Where it shows up:**
- **Machine translation:** English to French (different-length sequences)
- **Text summarization:** Long article to short summary
- **Speech recognition:** Audio waveform to text transcription
- **Chatbots:** User message to response
- **Image captioning:** (CNN encoder + RNN decoder)

**The big three architectures built on this:**

| Architecture | Encoder | Decoder | Example Model |
|-------------|---------|---------|---------------|
| **Encoder-only** | Yes | No | BERT (bidirectional understanding) |
| **Decoder-only** | No | Yes | GPT (autoregressive generation) |
| **Encoder-Decoder** | Yes | Yes | T5 (translation, summarization, Q&A) |

> **Interview Tip:** The progression from encoder-decoder to attention to transformers is one of the most important narratives in modern ML. Be able to explain: "The fixed-size context vector is a bottleneck for long sequences. Attention lets the decoder attend to all encoder positions. The Transformer removes recurrence entirely and uses only attention, enabling massive parallelization."

---

# Quick-Reference Formula Card

| Formula | Expression | Use |
|---------|-----------|-----|
| **Linear Regression** | y = w*X + b | Continuous prediction |
| **Sigmoid** | 1 / (1 + e^(-z)) | Logistic regression output, binary probability |
| **MSE** | (1/n) * sum((y - y_hat)^2) | Regression loss |
| **Binary Cross-Entropy** | -(1/n) * sum(y*log(p) + (1-y)*log(1-p)) | Classification loss |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many correct |
| **Recall** | TP / (TP + FN) | Of actual positives, how many caught |
| **F1** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| **L2 Penalty** | lambda * sum(w_i^2) | Ridge regularization |
| **L1 Penalty** | lambda * sum(\|w_i\|) | Lasso regularization (sparse) |
| **SGD Update** | w = w - lr * dL/dw | Weight update rule |
| **Adam** | w = w - lr * m_hat / (sqrt(v_hat) + eps) | Adaptive per-parameter update |
| **Conv Output Size** | floor((W - k + 2p) / s) + 1 | Spatial dimension after convolution |
| **Conv Layer Params** | C_out * (C_in * k * k + 1) | Parameter count for one conv layer |
| **FC Layer Params** | m * n + n | Parameters for a dense layer (m inputs, n outputs) |
| **R-squared** | 1 - (SS_res / SS_tot) | Variance explained by regression |
| **FPR** | FP / (FP + TN) | False Positive Rate (ROC x-axis) |
| **Gini Impurity** | 1 - sum(p_i^2) for each class i | Decision tree split criterion (0 = pure) |
| **Information Gain** | Entropy(parent) - weighted avg Entropy(children) | Entropy-based split criterion |
| **RF default max_features** | sqrt(n_features) for classification | Features considered per split in Random Forest |
