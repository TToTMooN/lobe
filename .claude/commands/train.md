Launch a training run for a VLA model.

Ask the user for:
1. **Model**: xvla, pi0.5, or walloss
2. **Dataset**: HuggingFace repo ID (e.g. yourname/yam-red-cube)
3. **Steps**: Number of training steps (default: 20000)
4. **Batch size**: (default: 8 for xvla, omit for others)
5. **Output dir**: checkpoint save path (default: checkpoints/<model>-<dataset-name>)

Then construct and run the appropriate `uv run lerobot-train` command:

- **xvla**: `--policy.path=lerobot/xvla-base --policy.dtype=bfloat16`
- **pi0.5**: `--policy.path=physical-intelligence/pi0.5 --policy.dtype=bfloat16`
- **walloss**: `--policy.type=wall_x --policy.path=x-square-robot/wall-oss-flow`
