"""Tests for FlowMatchingPolicy — verifies it matches DiffusionPolicy interface."""

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingModel, FlowMatchingPolicy


def _make_fake_stats(features: dict[str, PolicyFeature], norm_map: dict) -> dict:
    """Create fake dataset stats for normalization modules."""
    stats = {}
    for key, feat in features.items():
        stats[key] = {
            "min": torch.zeros(feat.shape),
            "max": torch.ones(feat.shape),
            "mean": torch.zeros(feat.shape),
            "std": torch.ones(feat.shape),
        }
    return stats


def _make_config(**overrides) -> FlowMatchingConfig:
    config = FlowMatchingConfig(**overrides)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(5,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    config.validate_features()
    return config


def _make_config_with_images(**overrides) -> FlowMatchingConfig:
    config = FlowMatchingConfig(**overrides)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    config.validate_features()
    return config


def _make_batch(config, batch_size=4):
    batch = {
        "observation.state": torch.randn(batch_size, config.n_obs_steps, config.robot_state_feature.shape[0]),
        "action": torch.randn(batch_size, config.horizon, config.action_feature.shape[0]),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }
    if config.env_state_feature:
        batch["observation.environment_state"] = torch.randn(
            batch_size, config.n_obs_steps, config.env_state_feature.shape[0]
        )
    if config.image_features:
        img_shape = next(iter(config.image_features.values())).shape
        n_cams = len(config.image_features)
        batch["observation.images"] = torch.randn(batch_size, config.n_obs_steps, n_cams, *img_shape)
    return batch


# ============================================================
# FlowMatchingModel tests (core math)
# ============================================================


def test_flow_matching_loss_shape():
    """compute_loss returns a scalar tensor."""
    config = _make_config()
    model = FlowMatchingModel(config)
    loss = model.compute_loss(_make_batch(config))
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.item() > 0


def test_flow_matching_inference_shape():
    """generate_actions returns (B, n_action_steps, action_dim)."""
    config = _make_config(num_inference_steps=2)
    model = FlowMatchingModel(config)
    model.eval()
    batch = _make_batch(config)
    batch.pop("action")
    batch.pop("action_is_pad")
    with torch.no_grad():
        actions = model.generate_actions(batch)
    assert actions.shape == (4, 8, 2), f"Expected (4, 8, 2), got {actions.shape}"


def test_single_step_inference():
    """1-step inference (the main advantage of flow matching)."""
    config = _make_config(num_inference_steps=1)
    model = FlowMatchingModel(config)
    model.eval()
    batch = _make_batch(config, batch_size=2)
    batch.pop("action")
    batch.pop("action_is_pad")
    with torch.no_grad():
        actions = model.generate_actions(batch)
    assert actions.shape == (2, 8, 2)


def test_training_step_gradients():
    """Full forward + backward pass produces gradients."""
    config = _make_config()
    model = FlowMatchingModel(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss = model.compute_loss(_make_batch(config))
    loss.backward()
    optimizer.step()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed"


def test_outputs_are_finite():
    """All outputs are non-NaN and non-Inf across different step counts."""
    config = _make_config()
    model = FlowMatchingModel(config)
    model.eval()
    batch = _make_batch(config)
    batch.pop("action")
    batch.pop("action_is_pad")

    for steps in [1, 4, 16]:
        model.num_inference_steps = steps
        with torch.no_grad():
            actions = model.generate_actions(batch)
        assert torch.isfinite(actions).all(), f"Non-finite outputs with {steps} steps"


def test_loss_decreases_with_training():
    """Loss should decrease over a few training steps (basic learning signal)."""
    config = _make_config()
    model = FlowMatchingModel(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    batch = _make_batch(config, batch_size=32)

    losses = []
    for _ in range(20):
        loss = model.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ============================================================
# FlowMatchingPolicy tests (full policy with normalization)
# ============================================================


def test_policy_forward():
    """Policy forward() returns (loss, None)."""
    config = _make_config()
    stats = _make_fake_stats({**config.input_features, **config.output_features}, config.normalization_mapping)
    policy = FlowMatchingPolicy(config, dataset_stats=stats)
    batch = _make_batch(config)
    loss, info = policy.forward(batch)
    assert loss.shape == ()
    assert info is None


def test_policy_select_action():
    """Policy select_action returns a single action."""
    config = _make_config()
    stats = _make_fake_stats({**config.input_features, **config.output_features}, config.normalization_mapping)
    policy = FlowMatchingPolicy(config, dataset_stats=stats)
    policy.eval()

    single_obs = {
        "observation.state": torch.randn(1, config.robot_state_feature.shape[0]),
        "observation.environment_state": torch.randn(1, config.env_state_feature.shape[0]),
    }

    policy.reset()
    with torch.no_grad():
        action = policy.select_action(single_obs)
    assert action.shape == (1, 2), f"Expected (1, 2), got {action.shape}"


def test_policy_action_chunking():
    """Policy caches action chunk and pops one at a time."""
    config = _make_config()
    stats = _make_fake_stats({**config.input_features, **config.output_features}, config.normalization_mapping)
    policy = FlowMatchingPolicy(config, dataset_stats=stats)
    policy.eval()

    single_obs = {
        "observation.state": torch.randn(1, config.robot_state_feature.shape[0]),
        "observation.environment_state": torch.randn(1, config.env_state_feature.shape[0]),
    }

    policy.reset()
    actions = []
    with torch.no_grad():
        for _ in range(config.n_action_steps):
            action = policy.select_action(single_obs)
            actions.append(action)

    assert len(actions) == config.n_action_steps
    # Actions within a chunk should differ (they come from different timesteps)
    assert not torch.allclose(actions[0], actions[-1])


# ============================================================
# FlowMatchingPolicy with images
# ============================================================


def test_policy_with_images_forward():
    """Policy forward works with image observations."""
    config = _make_config_with_images()
    stats = _make_fake_stats({**config.input_features, **config.output_features}, config.normalization_mapping)
    policy = FlowMatchingPolicy(config, dataset_stats=stats)
    # forward() expects individual image keys (not pre-stacked "observation.images")
    # because it normalizes first, then stacks
    batch_size = 4
    batch = {
        "observation.state": torch.randn(batch_size, config.n_obs_steps, 2),
        "observation.image": torch.randn(batch_size, config.n_obs_steps, 3, 96, 96),
        "action": torch.randn(batch_size, config.horizon, 2),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }
    loss, _ = policy.forward(batch)
    assert loss.shape == ()
    assert loss.item() > 0


def test_policy_with_images_inference():
    """Policy select_action works with image observations."""
    config = _make_config_with_images()
    stats = _make_fake_stats({**config.input_features, **config.output_features}, config.normalization_mapping)
    policy = FlowMatchingPolicy(config, dataset_stats=stats)
    policy.eval()

    single_obs = {
        "observation.state": torch.randn(1, 2),
        "observation.image": torch.randn(1, 3, 96, 96),
    }

    policy.reset()
    with torch.no_grad():
        action = policy.select_action(single_obs)
    assert action.shape == (1, 2)


# ============================================================
# Config validation tests
# ============================================================


def test_config_rejects_invalid_inference_steps():
    """num_inference_steps must be >= 1."""
    try:
        FlowMatchingConfig(num_inference_steps=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_config_rejects_negative_sigma():
    """sigma must be >= 0."""
    try:
        FlowMatchingConfig(sigma=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_config_horizon_must_match_downsampling():
    """horizon must be divisible by 2^len(down_dims)."""
    try:
        FlowMatchingConfig(horizon=15, down_dims=(512, 1024, 2048))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import sys

    tests = [
        test_flow_matching_loss_shape,
        test_flow_matching_inference_shape,
        test_single_step_inference,
        test_training_step_gradients,
        test_outputs_are_finite,
        test_loss_decreases_with_training,
        test_policy_forward,
        test_policy_select_action,
        test_policy_action_chunking,
        test_policy_with_images_forward,
        test_policy_with_images_inference,
        test_config_rejects_invalid_inference_steps,
        test_config_rejects_negative_sigma,
        test_config_horizon_must_match_downsampling,
    ]

    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed.append(test.__name__)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    print("All tests passed!")
