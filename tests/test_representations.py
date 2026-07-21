from pathlib import Path

import numpy as np
import pytest
import torch

from surfacelora.representations import (
    CacheMetadata,
    RepresentationCache,
    extract_representation,
    final_nonpadding_indices,
    lasttok,
    meanpool,
    validate_cache_pair,
    validate_cache_triplet,
)


def test_lasttok_supports_left_and_right_padding():
    hidden = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    left_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    right_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    assert final_nonpadding_indices(left_mask).tolist() == [4, 4]
    assert final_nonpadding_indices(right_mask).tolist() == [2, 3]
    assert torch.equal(lasttok(hidden, left_mask), torch.stack((hidden[0, 4], hidden[1, 4])))
    assert torch.equal(lasttok(hidden, right_mask), torch.stack((hidden[0, 2], hidden[1, 3])))


def test_meanpool_excludes_padding_and_layer_selection():
    embeddings = torch.full((2, 4, 1), -1.0)
    first_block = torch.zeros(2, 4, 1)
    final_block = torch.tensor([[[100.0], [2.0], [4.0], [6.0]], [[3.0], [5.0], [100.0], [100.0]]])
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    states = (embeddings, first_block, final_block)
    assert torch.allclose(meanpool(final_block, mask), torch.tensor([[4.0], [4.0]]))
    assert torch.allclose(extract_representation(states, mask, "meanpool", layer=-1), torch.tensor([[4.0], [4.0]]))
    assert torch.equal(extract_representation(states, mask, "lasttok", layer=0), torch.zeros(2, 1))


def _cache(split: str, offset: int) -> RepresentationCache:
    return RepresentationCache(
        vectors=np.ones((2, 3), dtype=np.float32),
        labels=np.asarray([0, 1]),
        example_indices=np.asarray([0, 1]),
        row_keys=np.asarray([f"key-{offset}", f"key-{offset + 1}"]),
        metadata=CacheMetadata(
            split=split,
            representation="lasttok",
            layer=-1,
            checkpoint_identifier="step-200",
            tokenizer_identifier="synthetic",
            prompt_identifier="synthetic-v1",
            maximum_sequence_length=16,
            padding_side="left",
            truncation_side="left",
            label_order=("A", "B"),
        ),
    )


def test_cache_roundtrip_and_cross_split_identity(tmp_path: Path):
    caches = [_cache("train", 0), _cache("validation", 2), _cache("test", 4)]
    path = tmp_path / "train.npz"
    caches[0].save(path)
    loaded = RepresentationCache.load(path)
    assert loaded.alignment_digest == loaded.digest()
    validate_cache_pair([loaded, caches[1]])
    validate_cache_triplet([loaded, caches[1], caches[2]])


def test_cache_rejects_bad_order_duplicates_and_metadata_mismatch():
    bad = _cache("train", 0)
    bad.example_indices = np.asarray([1, 0])
    with pytest.raises(ValueError, match="contiguous"):
        bad.validate()
    caches = [_cache("train", 0), _cache("validation", 2), _cache("test", 4)]
    caches[2].metadata = CacheMetadata(**{**caches[2].metadata.__dict__, "checkpoint_identifier": "other"})
    with pytest.raises(ValueError, match="checkpoint_identifier"):
        validate_cache_triplet(caches)
