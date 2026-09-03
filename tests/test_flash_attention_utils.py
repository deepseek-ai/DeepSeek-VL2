"""Regression tests for the FlashAttention padding helpers."""

import torch

from deepseek_vl2.models.modeling_deepseek import _get_unpad_data


def test_get_unpad_data_returns_ragged_indices_and_offsets():
    """Ragged masks should produce int32 cumulative lengths for FlashAttention."""
    attention_mask = torch.tensor(
        [[True, True, False, False], [False, True, True, True]],
        dtype=torch.bool,
    )

    indices, cumulative_lengths, max_length = _get_unpad_data(attention_mask)

    assert indices.tolist() == [0, 1, 5, 6, 7]
    assert cumulative_lengths.dtype == torch.int32
    assert cumulative_lengths.tolist() == [0, 2, 5]
    assert max_length == 3


def test_get_unpad_data_accepts_integer_masks():
    """The helper should preserve the same layout for the 2-D masks used by FA2."""
    attention_mask = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.long)

    indices, cumulative_lengths, max_length = _get_unpad_data(attention_mask)

    assert indices.tolist() == [0, 2, 5]
    assert cumulative_lengths.tolist() == [0, 2, 3]
    assert max_length == 2
