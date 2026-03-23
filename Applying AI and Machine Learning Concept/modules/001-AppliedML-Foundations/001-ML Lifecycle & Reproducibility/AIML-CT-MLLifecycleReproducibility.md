# ML Lifecycle & Reproducibility

**Estimated Time:** 10 Minutes

## Introduction

Welcome to Applied Machine Learning Foundations. Before you dive into algorithms and model choices, it helps to see the whole journey from raw information to something that runs reliably in the real world. Think of a machine learning project like opening a restaurant: you need dependable ingredients (data), a tested recipe (training and evaluation), a kitchen that can repeat the same dish every night (reproducibility), and a service window where customers actually get their meals (deployment). If any step is vague or inconsistent, the team cannot debug problems, compare improvements fairly, or trust what they ship.

This lesson explains the standard machine learning lifecycle from data ingestion through deployment and introduces reproducibility habits you will use throughout the module. The goal is clarity on **why** each stage exists and **when** to lean on practices like fixed random seeds and version control. Detailed implementation belongs in labs, notebooks, and the live lecture.

## Core Concepts

### The standard ML lifecycle

Machine learning work is rarely a single script that runs once. Most teams follow a loop with recognizable stages. Names vary by company, but the ideas are consistent.

**Problem framing and success criteria.** You clarify the business or research question, constraints (latency, fairness, cost), and how you will measure success before you touch data. Without this, you risk building an accurate model for the wrong task.

**Data ingestion.** Raw data arrives from databases, files, APIs, streams, or partners. Ingestion is not just "loading a CSV"; it includes understanding schemas, access patterns, and how often data refreshes. The point is to bring information into a controlled place where you can inspect and version it alongside your code.

**Exploratory data analysis (EDA) and data cleaning.** You profile distributions, missing values, duplicates, and outliers. You decide what to fix, drop, or flag. EDA with **pandas** and visualization with **matplotlib** or **seaborn** is where intuition forms; Jupyter notebooks are a natural home here because you are still discovering, not packaging final logic.

**Feature engineering and preprocessing.** You transform raw fields into inputs the model can use: scaling, encoding categories, deriving ratios, handling time zones, and more. This step encodes domain knowledge. Choices made here strongly influence outcomes, so they belong in the same "contract" as the model itself.

**Model selection and training.** You choose candidate algorithms (often starting with **scikit-learn** estimators), split data responsibly, and fit models. Training is where randomness enters: shuffling, initialization, subsampling, and many algorithms' internal random draws.

**Evaluation.** You measure performance on held-out data or cross-validation folds using metrics aligned to your problem (accuracy alone is often insufficient). Evaluation tells you whether changes actually help or whether you got lucky on one split.

**Deployment and monitoring.** A trained model becomes useful when it is integrated into applications or batch pipelines, with logging, latency targets, and checks for data drift or degraded performance. Deployment closes the loop: new failures or shifting data send you back to earlier stages.

**Iteration.** The lifecycle repeats. New data, features, constraints, or bugs send you back to EDA, training, or even reframing the problem. Treating ML as a cycle rather than a straight line keeps expectations realistic.

Together, these steps answer the question: how does work move from "we have data somewhere" to "we have a system we can run and improve with confidence?"

### Reproducibility: why it matters

Reproducibility means that you, a teammate, or an automated job can rerun an analysis or training job and get the **same** results given the same inputs. That matters for debugging ("did my fix work?"), for science and compliance ("can we show how this decision was made?"), and for production ("will tonight's training match yesterday's?").

Two pillars show up in almost every Python ML stack: **controlling randomness** and **controlling versions**.

**Random seeds.** Many steps draw random numbers: train/test splits, bootstrap samples, weight initialization, and algorithms that use randomness internally. If seeds drift, two runs of the "same" notebook produce different metrics, which makes fair comparison impossible.

Setting seeds does not remove all randomness from the world, but it fixes the pseudo-random sequences your code uses. In practice you often set seeds for Python's `random` module, **NumPy**, and library-specific knobs such as `random_state` in **scikit-learn**:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Use the **same** documented seed across related experiments when you want comparable runs; change the seed when you deliberately want to test stability across multiple random trials.

**Version control with Git.** Your repository is the memory of the project. Commits capture **code** changes; in mature workflows you also track or document **data snapshots**, **configuration**, and **environment** so a run is replayable. Version control answers: "Which script and which settings produced this model file?"

Typical habits include small, focused commits, meaningful messages, and branches for experiments:

```bash
git add train_model.py config.yaml
git commit -m "Fix preprocessing: median imputation for numeric nulls"
```

Pair Git with disciplined notes on data sources and parameters (even a short README or experiment log) so people are not guessing which CSV or which hyperparameters matched a given result.

**Notebooks versus `.py` files.** Jupyter notebooks excel for EDA and narrative exploration. Moving stable preprocessing and training steps into **importable `.py` modules** reduces copy-paste errors and makes Git diffs readable. You will still orchestrate experiments in notebooks sometimes; the lifecycle idea is to promote what works into reusable code when it stops being exploratory.

### When to emphasize each practice

Lean on **clear lifecycle stages** when scope creeps, metrics argue with each other, or nobody knows where a bug entered the pipeline. Lean on **seeds** when you need comparable metrics between two ideas on the same data. Lean on **Git** whenever more than one person touches the project or when you might need to roll back. These habits are not bureaucracy; they are how teams keep velocity without losing trust in results.

## Connecting to Practice

In this curriculum you will work in **Python 3.10+** with **NumPy**, **pandas**, and **SciPy** for numerical work; **scikit-learn** for modeling; and **matplotlib** and **seaborn** for plots. Use notebooks to explore and visualize, and extract repeatable steps into `.py` files as your understanding solidifies. When you train estimators, pass explicit `random_state` values and keep your splits consistent while you compare approaches. When you change preprocessing or features, commit those changes so your teammates (and your future self) can trace what the model actually saw.

You do not need to master every deployment pattern in this first reading. You do need the mental map: data in, quality understood, features defined, model trained and evaluated honestly, then delivered and watched. Reproducibility is the thread that ties those stages into a story others can verify.

---

## Further Learning & Resources

**Documentation and reading**

- **[scikit-learn User Guide: Model persistence](https://scikit-learn.org/stable/model_persistence.html)** - *Docs*: Covers saving and loading trained models consistently across environments.
- **[NumPy: Random sampling](https://numpy.org/doc/stable/reference/random/index.html)** - *Docs*: Explains how NumPy manages random number generation in modern workflows.
- **[Pro Git book (online)](https://git-scm.com/book/en/v2)** - *Docs*: Practical deep dive into version control concepts and workflows.

**Interactive practice**

- **[Kaggle: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Short, hands-on notebooks with exercises covering the ML workflow.
- **[scikit-learn: Examples gallery](https://scikit-learn.org/stable/auto_examples/index.html)** - *Interactive*: Runnable examples you can adapt in your own environment.
- **[Google Colab](https://colab.research.google.com/)** - *Interactive*: Zero-setup notebooks to experiment with Python ML stacks in the browser.
