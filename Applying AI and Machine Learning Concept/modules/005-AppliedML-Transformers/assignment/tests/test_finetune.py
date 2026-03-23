"""Tests for architecture selection, pipeline design, fine-tuning strategy,
and attention complexity analysis."""

import pytest

from src.finetune import (
    select_architecture,
    order_finetuning_steps,
    recommend_finetuning_strategy,
    analyze_attention_complexity,
    CANONICAL_PIPELINE_STEPS,
)


class TestSelectArchitecture:
    """Tests for select_architecture."""

    @pytest.mark.parametrize("task,expected_arch", [
        ("sentiment_classification", "BERT"),
        ("named_entity_recognition", "BERT"),
        ("question_answering", "BERT"),
        ("spam_detection", "BERT"),
    ])
    def test_bert_tasks(self, task, expected_arch):
        result = select_architecture(task)
        assert result["architecture"] == expected_arch
        assert result["type"] == "encoder-only"

    @pytest.mark.parametrize("task,expected_arch", [
        ("text_generation", "GPT"),
        ("code_generation", "GPT"),
        ("story_continuation", "GPT"),
    ])
    def test_gpt_tasks(self, task, expected_arch):
        result = select_architecture(task)
        assert result["architecture"] == expected_arch
        assert result["type"] == "decoder-only"

    @pytest.mark.parametrize("task,expected_arch", [
        ("machine_translation", "T5"),
        ("text_summarization", "T5"),
        ("paraphrase_generation", "T5"),
    ])
    def test_t5_tasks(self, task, expected_arch):
        result = select_architecture(task)
        assert result["architecture"] == expected_arch
        assert result["type"] == "encoder-decoder"

    def test_returns_reasoning(self):
        result = select_architecture("sentiment_classification")
        assert "reasoning" in result
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 10

    def test_returns_all_keys(self):
        result = select_architecture("text_generation")
        assert {"architecture", "type", "reasoning"}.issubset(result.keys())


class TestOrderFinetuningSteps:
    """Tests for order_finetuning_steps."""

    def test_full_pipeline_order(self):
        import random
        shuffled = CANONICAL_PIPELINE_STEPS.copy()
        random.seed(42)
        random.shuffle(shuffled)
        result = order_finetuning_steps(shuffled)
        assert result == CANONICAL_PIPELINE_STEPS

    def test_subset_preserves_relative_order(self):
        subset = ["train_model", "load_tokenizer", "evaluate_on_test"]
        result = order_finetuning_steps(subset)
        assert result == ["load_tokenizer", "train_model", "evaluate_on_test"]

    def test_single_step(self):
        result = order_finetuning_steps(["train_model"])
        assert result == ["train_model"]

    def test_already_ordered(self):
        steps = ["select_pretrained_model", "load_tokenizer", "preprocess_dataset"]
        result = order_finetuning_steps(steps)
        assert result == steps

    def test_returns_same_elements(self):
        steps = ["evaluate_on_test", "train_model", "set_hyperparameters"]
        result = order_finetuning_steps(steps)
        assert set(result) == set(steps)
        assert len(result) == len(steps)


class TestRecommendFinetuningStrategy:
    """Tests for recommend_finetuning_strategy."""

    def test_low_compute_forces_feature_extraction(self):
        result = recommend_finetuning_strategy(10000, "low")
        assert result["strategy"] == "feature_extraction"
        assert result["freeze_base"] is True

    def test_small_dataset_feature_extraction(self):
        result = recommend_finetuning_strategy(500, "high")
        assert result["strategy"] == "feature_extraction"
        assert result["freeze_base"] is True
        assert result["learning_rate"] == 1e-3
        assert result["epochs"] == 10

    def test_medium_dataset_partial_finetuning(self):
        result = recommend_finetuning_strategy(3000, "medium")
        assert result["strategy"] == "partial_finetuning"
        assert result["freeze_base"] is False
        assert result["learning_rate"] == 2e-5
        assert result["epochs"] == 4

    def test_large_dataset_full_finetuning(self):
        result = recommend_finetuning_strategy(10000, "high")
        assert result["strategy"] == "full_finetuning"
        assert result["freeze_base"] is False
        assert result["learning_rate"] == 2e-5
        assert result["epochs"] == 3

    def test_boundary_1000(self):
        result = recommend_finetuning_strategy(1000, "medium")
        assert result["strategy"] == "partial_finetuning"

    def test_boundary_5000(self):
        result = recommend_finetuning_strategy(5000, "high")
        assert result["strategy"] == "full_finetuning"

    def test_returns_reasoning(self):
        result = recommend_finetuning_strategy(2000, "medium")
        assert "reasoning" in result
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 10

    def test_returns_all_keys(self):
        result = recommend_finetuning_strategy(1000, "medium")
        required = {"strategy", "freeze_base", "learning_rate", "epochs", "reasoning"}
        assert required.issubset(result.keys())


class TestAnalyzeAttentionComplexity:
    """Tests for analyze_attention_complexity."""

    def test_dk_computation(self):
        result = analyze_attention_complexity(128, 512, 8)
        assert result["d_k"] == 64

    def test_qkv_projection_ops(self):
        n, d, h = 128, 512, 8
        result = analyze_attention_complexity(n, d, h)
        expected = 3 * n * d * d
        assert result["qkv_projection_ops"] == expected

    def test_attention_score_ops(self):
        n, d, h = 128, 512, 8
        d_k = d // h
        result = analyze_attention_complexity(n, d, h)
        expected = h * n * d_k * n
        assert result["attention_score_ops"] == expected

    def test_attention_output_ops(self):
        n, d, h = 128, 512, 8
        d_k = d // h
        result = analyze_attention_complexity(n, d, h)
        expected = h * n * n * d_k
        assert result["attention_output_ops"] == expected

    def test_output_projection_ops(self):
        n, d, h = 128, 512, 8
        result = analyze_attention_complexity(n, d, h)
        expected = n * d * d
        assert result["output_projection_ops"] == expected

    def test_total_ops(self):
        n, d, h = 128, 512, 8
        result = analyze_attention_complexity(n, d, h)
        expected = (result["qkv_projection_ops"]
                    + result["attention_score_ops"]
                    + result["attention_output_ops"]
                    + result["output_projection_ops"])
        assert result["total_ops"] == expected

    def test_memory_for_scores(self):
        n, d, h = 128, 512, 8
        result = analyze_attention_complexity(n, d, h)
        expected = h * n * n
        assert result["memory_for_scores"] == expected

    def test_quadratic_flag(self):
        result = analyze_attention_complexity(64, 256, 4)
        assert result["quadratic_in_sequence_length"] is True

    def test_score_ops_scale_quadratically(self):
        r1 = analyze_attention_complexity(100, 512, 8)
        r2 = analyze_attention_complexity(200, 512, 8)
        ratio = r2["attention_score_ops"] / r1["attention_score_ops"]
        assert abs(ratio - 4.0) < 0.01, \
            "Doubling seq_length should quadruple attention score ops"

    def test_returns_all_keys(self):
        result = analyze_attention_complexity(10, 64, 4)
        required = {"d_k", "qkv_projection_ops", "attention_score_ops",
                     "attention_output_ops", "output_projection_ops",
                     "total_ops", "memory_for_scores",
                     "quadratic_in_sequence_length"}
        assert required.issubset(result.keys())
