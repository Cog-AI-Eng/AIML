"""Tests for CNN conceptual exercises."""

import pytest
import numpy as np

from src.cnn_concepts import (
    conv_output_size,
    conv_layer_params,
    pooling_output_size,
    trace_cnn_forward,
    compute_receptive_field,
    total_network_params,
)


class TestConvOutputSize:
    """Tests for conv_output_size."""

    def test_same_padding_stride_1(self):
        assert conv_output_size(32, 3, stride=1, padding=1) == 32

    def test_no_padding_stride_1(self):
        assert conv_output_size(32, 3, stride=1, padding=0) == 30

    def test_stride_2_no_padding(self):
        assert conv_output_size(32, 3, stride=2, padding=0) == 15

    def test_stride_2_with_padding(self):
        assert conv_output_size(32, 3, stride=2, padding=1) == 16

    def test_large_kernel(self):
        assert conv_output_size(28, 5, stride=1, padding=0) == 24

    def test_large_kernel_with_padding(self):
        assert conv_output_size(28, 5, stride=1, padding=2) == 28

    def test_1x1_convolution(self):
        assert conv_output_size(32, 1, stride=1, padding=0) == 32

    def test_kernel_equals_input(self):
        assert conv_output_size(7, 7, stride=1, padding=0) == 1

    def test_7x7_stride_2(self):
        assert conv_output_size(224, 7, stride=2, padding=3) == 112


class TestConvLayerParams:
    """Tests for conv_layer_params."""

    def test_first_layer_3x3(self):
        assert conv_layer_params(3, 32, 3, bias=True) == 896

    def test_first_layer_no_bias(self):
        assert conv_layer_params(3, 32, 3, bias=False) == 864

    def test_deeper_layer(self):
        assert conv_layer_params(32, 64, 3, bias=True) == 18496

    def test_1x1_conv(self):
        assert conv_layer_params(64, 128, 1, bias=True) == 8320

    def test_5x5_conv(self):
        assert conv_layer_params(3, 16, 5, bias=True) == 1216

    def test_large_layer(self):
        assert conv_layer_params(64, 128, 3, bias=True) == 73856

    def test_bias_difference(self):
        with_bias = conv_layer_params(32, 64, 3, bias=True)
        without_bias = conv_layer_params(32, 64, 3, bias=False)
        assert with_bias - without_bias == 64


class TestPoolingOutputSize:
    """Tests for pooling_output_size."""

    def test_standard_2x2_pool(self):
        assert pooling_output_size(32, 2) == 16

    def test_stride_defaults_to_pool_size(self):
        assert pooling_output_size(32, 2, stride=None) == 16

    def test_explicit_stride_equals_pool(self):
        assert pooling_output_size(32, 2, stride=2) == 16

    def test_3x3_pool(self):
        assert pooling_output_size(32, 3, stride=3) == 10

    def test_overlapping_pool(self):
        assert pooling_output_size(32, 3, stride=2) == 15

    def test_chain_pooling(self):
        size = pooling_output_size(32, 2)
        size = pooling_output_size(size, 2)
        size = pooling_output_size(size, 2)
        assert size == 4

    def test_odd_input(self):
        assert pooling_output_size(7, 2, stride=2) == 3


class TestTraceCnnForward:
    """Tests for trace_cnn_forward."""

    def test_single_conv_layer(self):
        shapes = trace_cnn_forward(
            (3, 32, 32),
            [{"type": "conv", "out_channels": 16, "kernel_size": 3, "padding": 1}]
        )
        assert shapes == [(16, 32, 32)]

    def test_conv_then_pool(self):
        shapes = trace_cnn_forward(
            (3, 32, 32),
            [
                {"type": "conv", "out_channels": 32, "kernel_size": 3, "padding": 1},
                {"type": "pool", "pool_size": 2},
            ]
        )
        assert shapes == [(32, 32, 32), (32, 16, 16)]

    def test_full_cifar_cnn(self):
        """Trace the CNN architecture from the original assignment."""
        layers = [
            {"type": "conv", "out_channels": 32, "kernel_size": 3, "padding": 1},
            {"type": "pool", "pool_size": 2},
            {"type": "conv", "out_channels": 64, "kernel_size": 3, "padding": 1},
            {"type": "pool", "pool_size": 2},
            {"type": "conv", "out_channels": 128, "kernel_size": 3, "padding": 1},
            {"type": "pool", "pool_size": 2},
            {"type": "flatten"},
            {"type": "linear", "out_features": 256},
            {"type": "linear", "out_features": 10},
        ]
        shapes = trace_cnn_forward((3, 32, 32), layers)
        expected = [
            (32, 32, 32),
            (32, 16, 16),
            (64, 16, 16),
            (64, 8, 8),
            (128, 8, 8),
            (128, 4, 4),
            (2048,),
            (256,),
            (10,),
        ]
        assert shapes == expected

    def test_strided_conv(self):
        shapes = trace_cnn_forward(
            (3, 224, 224),
            [{"type": "conv", "out_channels": 64, "kernel_size": 7,
              "stride": 2, "padding": 3}]
        )
        assert shapes == [(64, 112, 112)]

    def test_flatten_computes_product(self):
        shapes = trace_cnn_forward(
            (128, 4, 4),
            [{"type": "flatten"}]
        )
        assert shapes == [(2048,)]

    def test_conv_default_stride_and_padding(self):
        shapes = trace_cnn_forward(
            (3, 32, 32),
            [{"type": "conv", "out_channels": 16, "kernel_size": 3}]
        )
        assert shapes == [(16, 30, 30)]


class TestComputeReceptiveField:
    """Tests for compute_receptive_field."""

    def test_single_3x3_conv(self):
        layers = [{"kernel_size": 3, "stride": 1}]
        assert compute_receptive_field(layers) == 3

    def test_two_3x3_convs(self):
        layers = [
            {"kernel_size": 3, "stride": 1},
            {"kernel_size": 3, "stride": 1},
        ]
        assert compute_receptive_field(layers) == 5

    def test_conv_then_pool(self):
        layers = [
            {"kernel_size": 3, "stride": 1},
            {"kernel_size": 2, "stride": 2},
        ]
        assert compute_receptive_field(layers) == 4

    def test_cifar_cnn_architecture(self):
        """Receptive field for the 3-block conv-pool CNN."""
        layers = [
            {"kernel_size": 3, "stride": 1},   # conv1
            {"kernel_size": 2, "stride": 2},   # pool1
            {"kernel_size": 3, "stride": 1},   # conv2
            {"kernel_size": 2, "stride": 2},   # pool2
            {"kernel_size": 3, "stride": 1},   # conv3
            {"kernel_size": 2, "stride": 2},   # pool3
        ]
        assert compute_receptive_field(layers) == 22

    def test_single_1x1_conv(self):
        layers = [{"kernel_size": 1, "stride": 1}]
        assert compute_receptive_field(layers) == 1

    def test_strided_conv(self):
        layers = [{"kernel_size": 7, "stride": 2}]
        assert compute_receptive_field(layers) == 7

    def test_empty_layers(self):
        assert compute_receptive_field([]) == 1


class TestTotalNetworkParams:
    """Tests for total_network_params."""

    def test_single_conv_layer(self):
        conv = [{"in_channels": 3, "out_channels": 32,
                 "kernel_size": 3, "bias": True}]
        assert total_network_params(conv, []) == 896

    def test_single_linear_layer(self):
        linear = [{"in_features": 256, "out_features": 10, "bias": True}]
        assert total_network_params([], linear) == 2570

    def test_linear_no_bias(self):
        linear = [{"in_features": 256, "out_features": 10, "bias": False}]
        assert total_network_params([], linear) == 2560

    def test_full_cifar_cnn(self):
        """Total parameters for the 3-block CNN with classifier head."""
        conv = [
            {"in_channels": 3, "out_channels": 32,
             "kernel_size": 3, "bias": True},
            {"in_channels": 32, "out_channels": 64,
             "kernel_size": 3, "bias": True},
            {"in_channels": 64, "out_channels": 128,
             "kernel_size": 3, "bias": True},
        ]
        linear = [
            {"in_features": 2048, "out_features": 256, "bias": True},
            {"in_features": 256, "out_features": 10, "bias": True},
        ]
        total = total_network_params(conv, linear)
        expected = 896 + 18496 + 73856 + 524544 + 2570
        assert total == expected

    def test_bias_default_true(self):
        conv = [{"in_channels": 3, "out_channels": 32, "kernel_size": 3}]
        assert total_network_params(conv, []) == 896

    def test_mixed_bias(self):
        conv = [
            {"in_channels": 3, "out_channels": 32,
             "kernel_size": 3, "bias": True},
            {"in_channels": 32, "out_channels": 64,
             "kernel_size": 3, "bias": False},
        ]
        expected = 896 + 18432
        assert total_network_params(conv, []) == expected
