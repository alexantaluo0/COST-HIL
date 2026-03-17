#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for geometric augmentation with flip-action consistency."""

import torch

from lerobot.rl.augmentation import (
    GeometricAugmentationConfig,
    apply_action_augmentation,
    apply_geometric_augmentation,
    sample_augmentation_params,
)


def test_flip_action_consistency_double_flip_restores_original():
    """Flipping twice should restore the original action (flip is self-inverse)."""
    torch.manual_seed(42)
    config = GeometricAugmentationConfig(flip_prob=1.0, horizontal_axis=0)
    batch_size = 4
    actions = torch.randn(batch_size, 3)
    params = sample_augmentation_params(batch_size, 64, 64, config, actions.device)

    # Apply flip twice
    once = apply_action_augmentation(actions, params, config.horizontal_axis)
    twice = apply_action_augmentation(once, params, config.horizontal_axis)

    torch.testing.assert_close(twice, actions, msg="Double flip should restore original actions")


def test_flip_action_negates_horizontal_axis_only():
    """When flip=True, only horizontal_axis component should be negated."""
    config = GeometricAugmentationConfig(flip_prob=1.0, horizontal_axis=0)
    batch_size = 2
    actions = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    params = {
        "flip": torch.tensor([True, True]),
        "crop_top": torch.zeros(2, dtype=torch.long),
        "crop_left": torch.zeros(2, dtype=torch.long),
        "crop_h": 64,
        "crop_w": 64,
    }

    result = apply_action_augmentation(actions, params, config.horizontal_axis)

    expected = torch.tensor([[-1.0, 2.0, 3.0], [1.0, -2.0, -3.0]])
    torch.testing.assert_close(result, expected)


def test_flip_action_no_change_when_flip_false():
    """When flip=False for all samples, actions should be unchanged."""
    config = GeometricAugmentationConfig(flip_prob=0.0, horizontal_axis=0)
    batch_size = 4
    actions = torch.randn(batch_size, 3)
    params = sample_augmentation_params(batch_size, 64, 64, config, actions.device)

    result = apply_action_augmentation(actions, params, config.horizontal_axis)

    torch.testing.assert_close(result, actions)


def test_apply_geometric_augmentation_image_action_sync():
    """apply_geometric_augmentation should apply same params to all image keys and sync action."""
    torch.manual_seed(123)
    config = GeometricAugmentationConfig(flip_prob=0.5, crop_ratio_range=(1.0, 1.0), target_size=(64, 64))
    batch_size = 2
    H, W = 128, 128

    batch_state = {
        "observation.images.front": torch.rand(batch_size, 3, H, W),
        "observation.images.wrist": torch.rand(batch_size, 3, H, W),
    }
    batch_next_state = {
        "observation.images.front": torch.rand(batch_size, 3, H, W),
        "observation.images.wrist": torch.rand(batch_size, 3, H, W),
    }
    batch_actions = torch.randn(batch_size, 3)
    image_keys = ["observation.images.front", "observation.images.wrist"]

    new_state, new_next_state, new_actions = apply_geometric_augmentation(
        batch_state, batch_next_state, batch_actions, image_keys, config
    )

    assert new_state.keys() == batch_state.keys()
    assert new_next_state.keys() == batch_next_state.keys()
    assert new_state["observation.images.front"].shape == (batch_size, 3, 64, 64)
    assert new_next_state["observation.images.front"].shape == (batch_size, 3, 64, 64)
    assert new_actions.shape == batch_actions.shape


def test_enable_flip_false_no_action_change():
    """When enable_flip=False, actions should never be modified."""
    config = GeometricAugmentationConfig(enable_flip=False, flip_prob=1.0)
    batch_size = 4
    actions = torch.randn(batch_size, 3)
    params = sample_augmentation_params(batch_size, 64, 64, config, actions.device)
    result = apply_action_augmentation(actions, params, config.horizontal_axis)
    torch.testing.assert_close(result, actions)
    assert not params["flip"].any(), "flip should be all False when enable_flip=False"


def test_enable_crop_false_full_image():
    """When enable_crop=False, crop should use full image (crop_h=H, crop_w=W)."""
    config = GeometricAugmentationConfig(enable_crop=False, crop_ratio_range=(0.5, 0.5))
    batch_size = 2
    H, W = 128, 128
    params = sample_augmentation_params(batch_size, H, W, config, torch.device("cpu"))
    assert params["crop_h"] == H
    assert params["crop_w"] == W
    assert (params["crop_top"] == 0).all()
    assert (params["crop_left"] == 0).all()


def test_apply_geometric_augmentation_empty_image_keys():
    """When image_keys is empty, state/action should be returned unchanged."""
    batch_state = {"observation.state": torch.randn(2, 10)}
    batch_next_state = {"observation.state": torch.randn(2, 10)}
    batch_actions = torch.randn(2, 3)
    config = GeometricAugmentationConfig()

    new_state, new_next_state, new_actions = apply_geometric_augmentation(
        batch_state, batch_next_state, batch_actions, [], config
    )

    torch.testing.assert_close(new_state["observation.state"], batch_state["observation.state"])
    torch.testing.assert_close(new_next_state["observation.state"], batch_next_state["observation.state"])
    torch.testing.assert_close(new_actions, batch_actions)
