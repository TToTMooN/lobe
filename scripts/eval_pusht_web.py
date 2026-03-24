"""PushT Web Eval — browser-based policy viewer over HTTP.

Streams live policy rollouts as MJPEG to any browser. Works over SSH
without X11 forwarding. Open http://localhost:8080 after launching.

Usage:
    uv run python scripts/eval_pusht_web.py --checkpoint checkpoints/pusht_v10_crop_ema/flow_matching_50000
    uv run python scripts/eval_pusht_web.py --port 8080 --policy-type diffusion --checkpoint ...
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import gym_pusht  # noqa: F401
import gymnasium
import torch
import tyro
from flask import Flask, Response, render_template_string
from loguru import logger

import lobe.video_compat  # noqa: F401
from lobe import pusht

app = Flask(__name__)

# Shared state between policy thread and web server
state = {
    "frame": None,
    "info": {},
    "running": True,
    "reset": False,
    "paused": False,
}
state_lock = threading.Lock()


@dataclass
class Args:
    policy_type: str = "flow_matching"
    checkpoint: str = ""
    num_inference_steps: int = 10
    device: str = "cuda"
    dataset_repo_id: str = pusht.DEFAULT_DATASET
    port: int = 8080
    host: str = "0.0.0.0"
    max_steps: int = pusht.MAX_STEPS
    seed: int = 42
    render_size: int = 512


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LOBE — PushT Eval</title>
    <style>
        body { margin: 0; background: #1a1a2e; color: #eee; font-family: monospace;
               display: flex; flex-direction: column; align-items: center; padding: 20px; }
        h1 { color: #e94560; margin-bottom: 10px; }
        .stream { border: 2px solid #e94560; border-radius: 8px; }
        .controls { margin-top: 15px; display: flex; gap: 10px; }
        button { padding: 8px 20px; font-size: 14px; font-family: monospace; cursor: pointer;
                 background: #16213e; color: #eee; border: 1px solid #e94560; border-radius: 4px; }
        button:hover { background: #e94560; }
        .info { margin-top: 10px; font-size: 14px; color: #aaa; }
        #stats { margin-top: 5px; font-size: 13px; }
    </style>
</head>
<body>
    <h1>LOBE — PushT Policy Eval</h1>
    <img class="stream" src="/stream" width="{{ size }}" height="{{ size }}" />
    <div class="controls">
        <button onclick="fetch('/api/reset')">Reset (R)</button>
        <button onclick="fetch('/api/pause')">Pause/Resume (Space)</button>
    </div>
    <div class="info">{{ policy_type }} | {{ checkpoint }} | {{ inference_steps }} inference steps</div>
    <div id="stats"></div>
    <script>
        setInterval(async () => {
            const r = await fetch('/api/stats');
            const d = await r.json();
            document.getElementById('stats').innerText =
                `Episode ${d.episode} | Step ${d.step} | Reward ${d.reward.toFixed(3)} | ` +
                `Success ${d.success_rate.toFixed(0)}% | ${d.fps.toFixed(1)} FPS`;
        }, 500);
        document.addEventListener('keydown', e => {
            if (e.key === 'r') fetch('/api/reset');
            if (e.key === ' ') { e.preventDefault(); fetch('/api/pause'); }
        });
    </script>
</body>
</html>
"""


def draw_overlay(frame, info):
    """Draw HUD on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    texts = [
        f"Ep {info.get('episode', 0)} | Step {info.get('step', 0)}/{info.get('max_steps', 300)}",
        f"Reward: {info.get('reward', 0):.3f} | Success: {info.get('successes', 0)}/{info.get('episodes', 0)}",
        f"Latency: {info.get('latency_ms', 0):.1f}ms",
    ]
    for i, text in enumerate(texts):
        cv2.putText(overlay, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return overlay


def policy_loop(args: Args):
    """Background thread: runs policy rollouts and updates shared state."""
    logger.info("Loading policy...")
    dataset, features = pusht.load_dataset(args.dataset_repo_id)
    policy = pusht.create_policy(
        args.policy_type,
        features,
        dataset.meta.stats,
        num_inference_steps=args.num_inference_steps,
    )
    pusht.load_checkpoint(policy, args.checkpoint, args.device)
    policy.to(args.device)
    policy.eval()
    logger.info(f"Policy loaded: {sum(p.numel() for p in policy.parameters()):,} params")

    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    episode = 0
    total_successes = 0

    while state["running"]:
        obs, _ = env.reset(seed=args.seed + episode)
        policy.reset()
        episode += 1
        step = 0
        episode_reward = 0.0

        with state_lock:
            state["reset"] = False

        while step < args.max_steps and state["running"]:
            if state["paused"]:
                time.sleep(0.05)
                continue
            if state["reset"]:
                break

            t0 = time.perf_counter()
            batch = pusht.obs_to_batch(obs, args.device)
            with torch.no_grad():
                action = policy.select_action(batch)
            latency = (time.perf_counter() - t0) * 1000

            action_np = action[0].cpu().numpy().clip(0, 512)
            obs, reward, terminated, truncated, info = env.step(action_np)
            step += 1
            episode_reward += reward

            # Render frame
            frame = env.render()
            frame = cv2.resize(frame, (args.render_size, args.render_size), interpolation=cv2.INTER_NEAREST)

            info_dict = {
                "episode": episode,
                "step": step,
                "max_steps": args.max_steps,
                "reward": episode_reward / step,
                "latency_ms": latency,
                "successes": total_successes,
                "episodes": episode - 1,
            }
            frame = draw_overlay(frame, info_dict)

            with state_lock:
                state["frame"] = frame
                state["info"] = {
                    "episode": episode,
                    "step": step,
                    "reward": episode_reward / step,
                    "success_rate": (total_successes / max(episode - 1, 1)) * 100,
                    "fps": 1000.0 / max(latency, 1),
                }

            if terminated or truncated:
                if info.get("is_success", False):
                    total_successes += 1
                break

            time.sleep(max(0, 1.0 / pusht.FPS - (time.perf_counter() - t0)))

    env.close()


def generate_mjpeg():
    """Yield MJPEG frames for browser streaming."""
    while state["running"]:
        with state_lock:
            frame = state["frame"]
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(1.0 / 30)  # 30fps max stream rate


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        size=app.config["render_size"],
        policy_type=app.config["policy_type"],
        checkpoint=app.config["checkpoint"] or "random",
        inference_steps=app.config["inference_steps"],
    )


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def stats():
    import json

    with state_lock:
        return json.dumps(state.get("info", {"episode": 0, "step": 0, "reward": 0, "success_rate": 0, "fps": 0}))


@app.route("/api/reset")
def reset():
    with state_lock:
        state["reset"] = True
    return "ok"


@app.route("/api/pause")
def pause():
    with state_lock:
        state["paused"] = not state["paused"]
    return "ok"


def main():
    args = tyro.cli(Args)

    app.config["render_size"] = args.render_size
    app.config["policy_type"] = args.policy_type
    app.config["checkpoint"] = args.checkpoint
    app.config["inference_steps"] = args.num_inference_steps

    # Start policy loop in background thread
    policy_thread = threading.Thread(target=policy_loop, args=(args,), daemon=True)
    policy_thread.start()

    logger.info(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
