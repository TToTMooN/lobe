Add a new VLA/policy method to the LOBE codebase.

Given a method name (e.g. "octo", "rt-2", "diffusion-policy"), do the following:

1. **Literature search**: Search the web for the method's paper, published results on LIBERO/PushT/robomimic, recommended hyperparameters, and known issues.

2. **Code search**: Search GitHub for existing implementations — prioritize lerobot plugins, HuggingFace models, or official repos. Check if lerobot already has native support (`lerobot-train --policy.type=<name>`).

3. **Integration plan**: Based on findings, decide the integration path:
   - **If lerobot-native**: Add a preset to `scripts/train_vla.py` MODEL_PRESETS dict + config in `lobe/configs/`
   - **If external repo**: Add as dependency, create a wrapper in `lobe/policies/`
   - **If custom**: Implement in `lobe/policies/` following the FM policy pattern (modeling + configuration files)

4. **Implementation**: Add the method following the architecture in CLAUDE.md. Include:
   - Config dataclass in `lobe/configs/base.py`
   - Environment preset in `lobe/configs/<env>.py`
   - Factory registration in `lobe/policies/factory.py`

5. **Sanity test**: Run a quick training (100 steps) to verify the pipeline works: `uv run python scripts/train.py <preset> --train.steps 100`

6. **Log**: Add entry to experiments.tsv with the sanity test results.

Always check CLAUDE.md first principles — especially #1 (check existing implementations) and #8 (use existing tools).
