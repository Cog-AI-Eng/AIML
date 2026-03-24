"""
Transfer learning and fine-tuning conceptual exercises.

Associates demonstrate understanding of architecture selection,
pipeline design, training strategies, and computational complexity
for fine-tuning pre-trained transformer models.
No deep learning frameworks are required.
"""


VALID_TASKS = {
    "sentiment_classification",
    "text_generation",
    "machine_translation",
    "text_summarization",
    "named_entity_recognition",
    "code_generation",
    "question_answering",
    "paraphrase_generation",
    "story_continuation",
    "spam_detection",
}

CANONICAL_PIPELINE_STEPS = [
    "select_pretrained_model",
    "load_tokenizer",
    "preprocess_dataset",
    "split_train_validation",
    "configure_model_head",
    "set_hyperparameters",
    "train_model",
    "evaluate_on_validation",
    "select_best_checkpoint",
    "evaluate_on_test",
]


def select_architecture(task: str) -> dict:
    """Select the best transformer architecture for a given NLP task.

    Architecture mapping:
        BERT  (encoder-only)    -- understanding/classification tasks:
            sentiment_classification, named_entity_recognition,
            question_answering, spam_detection
        GPT   (decoder-only)   -- generative/completion tasks:
            text_generation, code_generation, story_continuation
        T5    (encoder-decoder) -- sequence-to-sequence tasks:
            machine_translation, text_summarization, paraphrase_generation

    Args:
        task: One of the strings listed in VALID_TASKS.

    Returns:
        dict with:
            "architecture": one of "BERT", "GPT", "T5"
            "type": "encoder-only", "decoder-only", or "encoder-decoder"
            "reasoning": one-sentence explanation of why this architecture fits
    """
    # TODO: Map each task to its correct architecture and return the dict.
    #   Use the mapping described in the docstring.
    raise NotImplementedError("Implement select_architecture")


def order_finetuning_steps(steps: list) -> list:
    """Put fine-tuning pipeline steps in the correct execution order.

    The canonical order is defined in CANONICAL_PIPELINE_STEPS:
        1. select_pretrained_model
        2. load_tokenizer
        3. preprocess_dataset
        4. split_train_validation
        5. configure_model_head
        6. set_hyperparameters
        7. train_model
        8. evaluate_on_validation
        9. select_best_checkpoint
        10. evaluate_on_test

    Args:
        steps: A shuffled list containing some or all of these step strings.

    Returns:
        The same strings sorted into the correct execution order.
    """
    # TODO: Sort the input steps according to CANONICAL_PIPELINE_STEPS order.
    #   Only include steps that appear in the input list.
    raise NotImplementedError("Implement order_finetuning_steps")


def recommend_finetuning_strategy(
    dataset_size: int, compute_budget: str
) -> dict:
    """Recommend a fine-tuning strategy based on constraints.

    Decision rules:
        1. If compute_budget == "low":
           -> feature_extraction, freeze_base=True, lr=1e-3, epochs=10
        2. Else if dataset_size < 1000:
           -> feature_extraction, freeze_base=True, lr=1e-3, epochs=10
        3. Else if dataset_size < 5000:
           -> partial_finetuning, freeze_base=False, lr=2e-5, epochs=4
        4. Else (dataset_size >= 5000):
           -> full_finetuning, freeze_base=False, lr=2e-5, epochs=3

    Args:
        dataset_size: Number of labeled training examples.
        compute_budget: One of "low", "medium", "high".

    Returns:
        dict with:
            "strategy": "feature_extraction", "partial_finetuning", or "full_finetuning"
            "freeze_base": bool
            "learning_rate": float
            "epochs": int
            "reasoning": str -- one-sentence explanation
    """
    # TODO: Apply the decision rules above and return the dict.
    raise NotImplementedError("Implement recommend_finetuning_strategy")


def analyze_attention_complexity(
    seq_length: int, d_model: int, num_heads: int
) -> dict:
    """Analyze the computational complexity of self-attention.

    Count multiplications (not multiply-accumulate) for each stage,
    ignoring the batch dimension. For a matrix multiply of (m, k) @ (k, n),
    the multiplication count is m * k * n.

    Stages:
        - QKV projections: 3 matrices, each (seq_length, d_model) @ (d_model, d_model)
        - Attention scores: num_heads parallel (seq_length, d_k) @ (d_k, seq_length)
        - Attention output: num_heads parallel (seq_length, seq_length) @ (seq_length, d_k)
        - Output projection: (seq_length, d_model) @ (d_model, d_model)

    Args:
        seq_length: Sequence length (n).
        d_model: Model dimension (d).
        num_heads: Number of attention heads (h).

    Returns:
        dict with:
            "d_k": per-head dimension (d_model // num_heads)
            "qkv_projection_ops": int -- multiplications for Q, K, V projections
            "attention_score_ops": int -- multiplications for all heads' Q @ K^T
            "attention_output_ops": int -- multiplications for all heads' weights @ V
            "output_projection_ops": int -- multiplications for W_o projection
            "total_ops": int -- sum of all above
            "memory_for_scores": int -- elements to store all attention score matrices
            "quadratic_in_sequence_length": bool (True)
    """
    # TODO: Compute each value using the formulas above:
    #   d_k = d_model // num_heads
    #   qkv = 3 * seq_length * d_model * d_model
    #   scores = num_heads * seq_length * d_k * seq_length (= seq_length^2 * d_model)
    #   attn_out = num_heads * seq_length * seq_length * d_k (= seq_length^2 * d_model)
    #   out_proj = seq_length * d_model * d_model
    #   memory = num_heads * seq_length * seq_length
    raise NotImplementedError("Implement analyze_attention_complexity")
