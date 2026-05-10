Launch a training run for a VLA model.

Arguments: $ARGUMENTS (optional: model, dataset, steps, num-gpus)

If arguments are insufficient, ask the user for:
1. **Model**: smolvla, pi0 (VLA via lerobot-train), or pusht-fm, libero-fm (custom FM via train.py)
2. **Dataset**: HuggingFace repo ID (e.g. HuggingFaceVLA/libero)
3. **Steps**: Number of training steps (default: from model preset)
4. **Num GPUs**: How many GPUs to use (default: all available)

For VLA models (smolvla, pi0), use train_vla.py with multi-GPU:
```
uv run python scripts/train_vla.py --model <model> --dataset <dataset> --steps <steps> --batch-size <bs> --num-gpus <n>
```
Note: batch_size is PER GPU. With 8 GPUs and batch=8, effective batch = 64.

For FM/Diffusion baselines, use train.py:
```
uv run python scripts/train.py <preset> --performance.no-compile
```
Available presets: pusht-fm, libero-fm (run `--help` for all)

After training, log results to experiments.tsv and suggest running /eval.
