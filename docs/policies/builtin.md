# Built-in Policies (lerobot)

LOBE registers all lerobot built-in policies. Use them via `--policy.type=<name>`:

| Policy type | Class | Description |
|---|---|---|
| `act` | ACTConfig | Action Chunking with Transformers |
| `diffusion` | DiffusionConfig | Diffusion Policy (Chi et al. 2023) |
| `tdmpc` | TDMPCConfig | TD-MPC |
| `vqbet` | VQBeTConfig | VQ-BeT |
| `pi0` | PI0Config | π₀ (Physical Intelligence) |
| `pi0_fast` | PI0FastConfig | π₀-FAST |
| `pi05` | PI05Config | π₀.5 |
| `smolvla` | SmolVLAConfig | SmolVLA (HuggingFace) |
| `xvla` | XVLAConfig | X-VLA |
| `wall_x` | WallXConfig | Wall-X |
| `groot` | GrootConfig | NVIDIA GR00T |
| `flow_matching` | FlowMatchingConfig | **Custom (LOBE)** — see [Flow Matching](flow_matching.md) |

## Train any of them

```bash
lobe-train --policy.type=diffusion --dataset.repo_id=...
lobe-train --policy.type=act --dataset.repo_id=...
lobe-train --policy.path=lerobot/smolvla_base --dataset.repo_id=...    # pretrained
```

## Configuration

Each policy has its own config dataclass. View available options with `--help`:

```bash
lobe-train --policy.type=diffusion --help | grep policy\\.
```

Common settings:

| Flag | Description |
|---|---|
| `--policy.type=<name>` | Policy type (mutually exclusive with `--policy.path`) |
| `--policy.path=<repo_or_dir>` | Pretrained checkpoint to fine-tune |
| `--policy.repo_id=<name>` | Required for HF Hub publishing (set to anything) |
| `--policy.n_action_steps=N` | Actions to execute per inference call |
| `--policy.normalization_mapping` | Override normalization (rarely needed) |
