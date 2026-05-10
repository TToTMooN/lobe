# Training

## Single command, any policy

```bash
lobe-train \
  --policy.type=<policy_type> \
  --dataset.repo_id=<your_dataset> \
  --batch_size=<N> \
  --steps=<S> \
  --output_dir=<path>
```

Replace `<policy_type>` with one of the registered policies (`diffusion`, `act`, `tdmpc`, `vqbet`, `pi0`, `pi0_fast`, `pi05`, `smolvla`, `xvla`, `wall_x`, `groot`, `flow_matching`).

For pretrained checkpoints (VLAs), use `--policy.path` instead of `--policy.type`:

```bash
lobe-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=HuggingFaceVLA/libero \
  ...
```

## Multi-GPU

LOBE uses [Accelerate](https://huggingface.co/docs/accelerate) for multi-GPU training. The pattern:

```bash
uv run python -m accelerate.commands.launch \
  --num_processes 8 --multi_gpu --mixed_precision bf16 \
  $(which lobe-train) \
  --policy.type=diffusion \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --batch_size=8 \
  --steps=100000 \
  --output_dir=/path/to/output
```

The effective batch size is `batch_size * num_processes = 8 * 8 = 64`.

!!! warning "LR scaling"
    If you change the effective batch size from a published config, scale the LR proportionally (linear scaling rule). Two of our experiments failed because we 4× the batch without scaling LR.

## Pinned hyperparameters

For reproducibility, use these tested configs:

=== "SmolVLA on LIBERO"
    ```bash
    --policy.path=lerobot/smolvla_base
    --batch_size=8 --steps=100000          # eff batch 64 with 8 GPUs
    --optimizer.lr=1e-4
    --policy.empty_cameras=1
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
    ```
    Result: ~82% on LIBERO 4-suite, 4 hours on 8×H100.

=== "Diffusion Policy on LIBERO"
    ```bash
    --policy.type=diffusion
    --batch_size=8 --steps=100000          # match published 6.4M total samples
    --optimizer.lr=1e-4
    --policy.repo_id=diffusion-libero
    ```
    *Tuning needed; published is 72.4%.*

=== "Flow Matching on LIBERO"
    ```bash
    --policy.type=flow_matching
    --batch_size=8 --steps=100000
    --optimizer.lr=1e-4
    --policy.repo_id=fm-libero
    ```
    *Tuning needed; same architecture as DP, expects similar result.*

## Common gotchas

### "uv sync removed my deps"

Some optional deps (robosuite, gym, opencv) conflict with lerobot's nightly torch in dependency resolution. They are not declared in `pyproject.toml`. Reinstall after every `uv sync`:

```bash
uv pip install robosuite==1.4.1 bddl easydict matplotlib gym pyopengl pyopengl-accelerate \
    --index-strategy unsafe-best-match
```

### "Output directory already exists"

lerobot-train refuses to overwrite output dirs. Either delete them or pass a fresh `--output_dir`.

### "Cannot specify both --policy.path and --policy.type"

Use `--policy.path` for pretrained checkpoints (loads architecture from the checkpoint config), and `--policy.type` for training from scratch.

### Disk full

Training writes large checkpoints. Always point `--output_dir` to a large disk (we use `/mnt/localssd`). Set `HF_HOME`, `WANDB_DIR`, `TMPDIR` to the same SSD.
