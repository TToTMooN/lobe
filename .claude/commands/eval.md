Evaluate a trained policy checkpoint.

Arguments: $ARGUMENTS (checkpoint path, optional: env, task, n_episodes)

1. **Detect policy type** from the checkpoint:
   - If it's a lerobot/HF checkpoint (has `pretrained_model/` or `config.json`): use `lerobot-eval`
   - If it's a custom FM/diffusion checkpoint (has `model.pt`): use the env-specific evaluate function

2. **Run evaluation**:
   - For lerobot policies (SmolVLA, pi0, etc.):
     ```
     lerobot-eval --policy.path=<checkpoint> --env.type=libero --env.task=libero_10 --eval.n_episodes=10
     ```
     Add `--rename_map` if needed (check the training config for camera remapping).

   - For custom policies (FM, diffusion):
     Load the checkpoint and call the env's evaluate() function directly.

3. **Log results** to experiments.tsv with success_rate, avg_reward, and notes.

4. **Report** success rate vs published baseline (check CLAUDE.md for target numbers).

If no checkpoint path is given, find the most recent checkpoint in `checkpoints/`.
