# Supervised vs. Unsupervised Learning

**Estimated Time:** 10 Minutes

## Introduction

When people say a system is "learning from data," they rarely mean one single method. In practice, machine learning is organized into **paradigms**: different ways of using data to discover patterns or make decisions. The two you will use most often are **supervised** and **unsupervised** learning. A third paradigm, **reinforcement learning**, shows up in robotics, games, and recommendation systems; you only need a clear picture of how it differs from the first two.

Think of the difference like planning a trip:

- **Supervised learning** is like studying with an answer key. For many examples, you are told both the situation and the correct outcome, and the model learns to map situations to outcomes.
- **Unsupervised learning** is like sorting a box of photos with no labels. No one tells you what each picture is; you group them by similarity or structure.
- **Reinforcement learning** is like learning to play a game by trial and error: you take actions, get rewards or penalties, and gradually improve a policy.

This lesson focuses on **supervised versus unsupervised** learning: what each one assumes, when each one fits, and how they show up in a typical Python stack (for example **NumPy**, **pandas**, **scikit-learn**, **matplotlib**, and **seaborn**). Detailed model-building belongs in labs and lectures; here, the goal is **why** and **when**.

## Core Concepts

### Supervised learning: learning from labeled examples

In **supervised learning**, the training data includes **inputs** (features) and **known outputs** (labels or targets). The algorithm learns a function that predicts the label from the features.

**When it fits:** You have (or can obtain) trustworthy labels and a clear prediction goal: classify, regress, rank, or detect something specific.

**Why it works:** The model is explicitly optimized to match the labels, so success is measurable with metrics you choose (accuracy, error, etc.).

**Types of targets (high level):**

- **Classification:** labels are discrete categories (spam vs. not spam).
- **Regression:** targets are continuous numbers (house price, demand).

A tiny illustration of the *idea* in code -- features `X` and labels `y` passed together to a scikit-learn estimator:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

You are not expected to master every parameter here; notice only that **`fit` needs both `X` and `y`** because supervision means "tell me the right answers for these examples."

### Unsupervised learning: structure without labels

In **unsupervised learning**, you typically have **inputs only**. The algorithm looks for **structure**: clusters, low-dimensional summaries, anomalies, or associations. There is no single "correct answer" column provided during training in the same way as in supervised learning.

**When it fits:** Labels are missing, expensive, or ill-defined; you want to explore data, segment customers, reduce dimensions for visualization, or flag unusual records.

**Why it is different:** Evaluation is often **indirect** (interpretability, stability of clusters, downstream task performance) rather than one simple accuracy number against a label column.

Example of the *shape* of the task -- only `X`, no `y` for fitting:

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
```

Cluster labels here are **discovered group ids**, not ground-truth categories unless you later validate them against domain knowledge.

### Reinforcement learning (brief contrast)

**Reinforcement learning (RL)** does not rely on a static table of labeled (input, output) pairs like classical supervised learning. An **agent** chooses **actions** in an **environment**, receives **rewards** (or penalties), and learns a policy that maximizes long-term reward. It is "supervised" in a loose sense by reward signals, but those are not the same as row-by-row labels in a dataset.

**Mental model:** supervised = "here are the right answers for these cases"; unsupervised = "find structure in this pile of inputs"; reinforcement = "learn by interacting and optimizing cumulative reward."

## Connecting to Practice

**Choosing supervised:** Use it when the business question sounds like "predict this known quantity or category" and you can define labels (even if building them takes work). Data pipelines often use **pandas** for tables, **NumPy** for arrays, and **scikit-learn** for models and consistent train/test splits.

**Choosing unsupervised:** Use it when the question sounds like "what natural groups exist?" or "how can we compress or visualize this?" or "what looks weird?" Visualization with **matplotlib** or **seaborn** is especially helpful to sanity-check clusters or projections -- but treat plots as **evidence**, not proof, without domain checks.

**Common mistake:** Calling a problem "unsupervised" because labels are messy. If you still have a target concept (even noisy), you are often in a **supervised or weakly supervised** mindset; the lifecycle and reproducibility topics you covered earlier matter here.

**What comes next in your path:** A dedicated video will reinforce these ideas; later modules will walk through **how** to build and compare models (for example linear and logistic regression and an algorithm selection framework). This lesson sets the vocabulary so those sessions land faster.

---

## Further Learning & Resources

**Reading and documentation**

- **[Scikit-learn: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)** - *Docs*: Overview of supervised estimators and how they fit into the library.
- **[Scikit-learn: Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)** - *Docs*: Clustering, decomposition, outlier detection, and related methods.
- **[Google Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)** - *Docs*: Short definitions of terms you will see across supervised, unsupervised, and reinforcement settings.

**Interactive practice**

- **[TensorFlow Neural Network Playground](https://playground.tensorflow.org/)** - *Interactive*: Interactive visualization of how decision boundaries respond to data and architecture.
- **[Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Short, hands-on notebooks with exercises in a supervised flavor.
- **[scikit-learn examples gallery](https://scikit-learn.org/stable/auto_examples/index.html)** - *Interactive*: Runnable examples you can adapt in a local environment or notebook.
