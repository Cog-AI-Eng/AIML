# Optimization Strategies

**Estimated Time:** 10 Minutes

## Introduction

The previous topic established the anatomy of an HPO job: you define hyperparameter ranges, an objective metric, and resource limits. The critical question this topic answers is *how* SageMaker selects hyperparameter values for each trial. The choice of optimization strategy determines how efficiently the search explores the hyperparameter space -- a good strategy finds the optimal region faster, reducing the number of trials (and cost) needed.

SageMaker supports four optimization strategies, each with different trade-offs between exploration breadth, exploitation depth, and computational overhead.

## Core Concepts

### Random Search

The simplest strategy. For each trial, SageMaker independently samples random values from each hyperparameter range.

**Strengths:** Easy to parallelize (all trials are independent), no sequential dependency, good for initial exploration of a large search space.

**Weaknesses:** No learning between trials. Trial 15 does not benefit from the results of trials 1-14. Requires more total trials to find good configurations.

**When to use:** As a starting point when you have no prior knowledge about good hyperparameter ranges, or when you want maximum parallelism (set `MaxParallelTrainingJobs` equal to `MaxNumberOfTrainingJobs` to run everything at once).

### Bayesian Optimization

SageMaker's default and most commonly used strategy. Bayesian optimization builds a probabilistic model (a Gaussian Process) of the objective function as trials complete. It uses this model to select the next trial's hyperparameters by balancing:

- **Exploitation:** Sampling near regions that have produced good results so far.
- **Exploration:** Sampling in regions where the model is uncertain (to avoid missing better configurations).

**Strengths:** Learns from previous trials, converges faster than random search, and typically finds better configurations with fewer total trials.

**Weaknesses:** Inherently sequential -- each trial's selection depends on previous results. Running many parallel trials reduces the learning benefit because parallel trials are selected without knowledge of each other's results.

**When to use:** Default choice for most HPO jobs. Set `MaxParallelTrainingJobs` to a moderate number (2-5) to balance learning with wall-clock time. Higher parallelism degrades toward random search behavior.

### Grid Search

Evaluates every combination of hyperparameter values from a predefined grid. You specify exact values (not ranges) for each hyperparameter.

**Strengths:** Exhaustive coverage, deterministic, guaranteed to find the best combination within the grid.

**Weaknesses:** Combinatorial explosion -- a grid of 5 parameters with 4 values each produces 1,024 trials. Impractical for more than 2-3 hyperparameters.

**When to use:** When you have a small number of categorical hyperparameters to evaluate exhaustively (e.g., comparing 3 optimizers x 4 learning rates = 12 trials).

### Hyperband

A multi-fidelity strategy that runs many trials with small budgets (few training epochs or data samples) and progressively allocates more budget to promising configurations while stopping poor ones early.

**Process:**

1. Start a large number of trials with a small training budget (e.g., 5 epochs each).
2. Evaluate all trials and discard the bottom-performing fraction.
3. Increase the budget for surviving trials (e.g., 10 epochs).
4. Repeat until only a few trials remain, training with the full budget.

**Strengths:** Extremely efficient when trials can be meaningfully evaluated with partial budgets. Explores broadly and exploits deeply.

**Weaknesses:** Requires the objective metric to be available at intermediate training stages (not all algorithms support this). May prematurely stop trials that converge slowly.

**When to use:** Deep learning models where training for a few epochs gives a reasonable signal about final performance. Less useful for XGBoost-style boosting where each round adds capacity.

### Strategy comparison

| Factor | Random | Bayesian | Grid | Hyperband |
| :--- | :--- | :--- | :--- | :--- |
| Learning between trials | No | Yes | No | Partial (through promotion) |
| Parallelizability | Full | Limited | Full | Moderate |
| Total trials needed | High | Low-Medium | Exact (all combos) | Medium |
| Best for | Initial exploration | Default optimization | Small discrete spaces | Deep learning tuning |
| `MaxParallelTrainingJobs` | Equal to total | 2-5 recommended | Equal to total | Strategy-managed |

### Configuring strategy in the console

When creating an HPO job (**SageMaker > Training > Hyperparameter tuning jobs > Create**), the **Strategy** dropdown offers these four options. The rest of the job configuration is the same regardless of strategy, except:

- **Grid search** requires you to provide explicit values (not ranges) for each hyperparameter.
- **Hyperband** has an additional parameter for the minimum and maximum training budget (`MinResource`, `MaxResource`).

### Warm starting

SageMaker supports **warm start** for HPO jobs: you can create a new HPO job that uses the results of previous jobs as prior knowledge. This is useful when:

- You ran an initial HPO job and want to refine the search in a narrower range around the best result.
- You retrained on new data and want to re-tune starting from the previous best configuration.

Configure warm start by setting `WarmStartConfig` in the job definition and referencing the ARN of the parent HPO job. Only Bayesian and Random strategies support warm start.

## Connecting to Practice

This topic gives you the tools to choose the right optimization strategy for different model types and budgets. The next topic, *Parameter Ranges in Pipelines*, shows how to embed HPO as an automated step in a SageMaker Pipeline so tuning runs as part of your CI/CD workflow rather than as a manual console task. The module assignment will have you run the same tuning job with both Random and Bayesian strategies and compare convergence rates.

## Further Learning & Resources

**Documentation and reading**

- **[HPO Strategies](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)** - *Docs*: AWS documentation on all four strategies with configuration examples.
- **[Hyperband Paper (Li et al., 2018)](https://jmlr.org/papers/v18/16-558.html)** - *Reading*: The research paper describing the Hyperband algorithm.

**Interactive practice**

- **[Warm Start HPO Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/hyperparameter_tuning/xgboost_warm_start)** - *Interactive*: Sample notebook demonstrating warm-start HPO with XGBoost on SageMaker.
