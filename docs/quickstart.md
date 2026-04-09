# Quick Start

## Install

```bash
git clone <repo-url>
cd lobe
uv sync --index-strategy unsafe-best-match
```

For LIBERO simulation:

```bash
uv pip install robosuite==1.4.1 bddl easydict matplotlib gym pyopengl pyopengl-accelerate \
    --index-strategy unsafe-best-match
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
echo "/tmp/LIBERO" > .venv/lib/python3.*/site-packages/libero_path.pth
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<EOF
datasets: /mnt/localssd/sunlingfeng/libero/datasets
bddl_files: /tmp/LIBERO/libero/libero/bddl_files
init_states: /tmp/LIBERO/libero/libero/init_files
EOF
```

For headless GPU rendering (LIBERO eval):

```bash
sudo apt-get install -y libnvidia-gl-580-server  # match your driver version
# Or fall back to software rendering: sudo apt-get install libosmesa6
```

## Verify install

```bash
lobe-train --help        # should show the lerobot-train CLI
lobe-eval --help         # should show the lerobot-eval CLI
lobe-serve --help        # should show "lobe-serve configuration"
```

## Train your first policy

Reproduce SmolVLA on LIBERO using the paper config (4 hours on 8×H100):

```bash
export HF_HOME=/mnt/localssd/$USER/cache/huggingface  # use SSD if root disk is small

uv run python -m accelerate.commands.launch \
  --num_processes 8 --multi_gpu --mixed_precision bf16 \
  $(which lobe-train) \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --batch_size=8 --steps=100000 \
  --output_dir=/mnt/localssd/$USER/checkpoints/smolvla-libero \
  --num_workers=8 \
  --policy.repo_id=smolvla-libero \
  --save_checkpoint=true \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --policy.empty_cameras=1
```

## Evaluate

```bash
MUJOCO_GL=egl lobe-eval \
  --policy.path=/mnt/localssd/$USER/checkpoints/smolvla-libero/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=10 \
  --policy.n_action_steps=10 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

## Serve to a robot

```bash
lobe-serve \
  --checkpoint=/mnt/localssd/$USER/checkpoints/smolvla-libero/checkpoints/100000/pretrained_model \
  --port 8000
```

Then connect from limb's `WebSocketPolicyClient` on port 8000.
