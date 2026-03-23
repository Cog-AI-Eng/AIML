# Explainability & SHAP Values

**Estimated Time:** 10 Minutes

## Introduction

A model that scores well on a test set can still be **wrong for the wrong reasons** -- relying on a leaky feature, a proxy for protected attributes, or a spurious correlation. **Explainability** is the practice of making **how** the model uses inputs understandable to humans. **SHAP** (SHapley Additive exPlanations) is one rigorous way to attribute predictions to **features** using ideas borrowed from cooperative game theory: each feature "pays" its share of the outcome.

## Core Concepts

**Why explainability matters**

Stakeholders ask "why was this denied?" or "why is this tumor high-risk?" An accuracy number does not answer that. Explainability supports **debugging**, **fairness review**, **regulatory** narratives, and **trust** -- not blind trust, but **inspectable** trust.

**Black-box models**

Neural nets and complex ensembles can be opaque. You still need **global** stories (which features matter overall?) and **local** stories (why *this* prediction?). SHAP is often used for both via aggregations and per-row plots.

**SHAP values (conceptual)**

For one prediction, each feature gets a **contribution** such that contributions sum (with a baseline) to the model's output. Features that push the prediction toward "positive" get one sign; features that push the other way get the opposite. The **baseline** is the reference point (e.g., average prediction); SHAP explains deviations from that baseline.

**Shapley intuition**

Imagine splitting a payout among players who contributed unequally. Shapley values fairly assign credit by considering **all orderings** of players -- what each player adds when joining. Translated to ML: what does adding feature *j* do to the prediction, averaged across sensible coalitions of other features? That fairness property is why SHAP is popular for **consistent** attribution -- though it can be **expensive** to compute exactly for complex models, so approximations are common in practice.

**How to read feature importance**

A bar of "mean |SHAP|" shows **global** impact magnitude; a **beeswarm** or **waterfall** plot shows **direction** and spread. High magnitude means "this feature often moves the prediction a lot"; the sign tells you *how* it pushes. Always ask whether the feature is **legitimate** or a **proxy** -- explainability reveals *dependence*, not *causation*.

```python
# Pseudocode shape (actual SHAP usage covered in hands-on sessions):
# import shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
```

## Connecting to Practice

Use SHAP to **sanity-check** models before deployment: does the top feature match domain knowledge? If not, investigate data leakage or shortcut learning. Pair SHAP with **error analysis** (confusion matrix slices) so you do not only explain average behavior. Remember: explanations are **approximations** and depend on background data and explainer choice -- document those choices like any other evaluation assumption.

---

## Further Learning & Resources

**Documentation**

- **[SHAP documentation](https://shap.readthedocs.io/en/latest/)** - *Docs*: Official library documentation with API reference and tutorials.
- **[Scikit-learn: Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)** - *Docs*: A related global feature importance method built into sklearn.

**Interactive**

- **[Interpretable ML book: Shapley values chapter](https://christophm.github.io/interpretable-ml-book/shapley.html)** - *Interactive*: Free online textbook with clear visual explanations of Shapley values.
- **[SHAP GitHub (examples and notebooks)](https://github.com/shap/shap)** - *Interactive*: Runnable notebooks demonstrating SHAP on various model types.
