"""PushT Web Eval — browser-based policy viewer over HTTP.

Streams live policy rollouts as MJPEG to any browser. Works over SSH
without X11 forwarding. Open http://localhost:8080 after launching.

Usage:
    uv run python scripts/eval_pusht_web.py --checkpoint checkpoints/pusht_v10_crop_ema/flow_matching_50000
    uv run python scripts/eval_pusht_web.py --port 8080 --policy-type diffusion --checkpoint ...
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import cv2
import gym_pusht  # noqa: F401
import gymnasium
import torch
import tyro
from flask import Flask, Response, render_template_string, request
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
    "n_action_steps": 4,
    "num_inference_steps": 10,
    "rebuild_policy": False,
    # Mouse intervention
    "mouse_active": False,
    "mouse_pos": None,  # (x, y) in env coords [0, 512]
    "mode": "watch",  # watch | intervene
    "recorded_episodes": [],
    "current_recording": [],
}
state_lock = threading.Lock()


@dataclass
class Args:
    policy_type: str = "flow_matching"
    checkpoint: str = ""
    num_inference_steps: int = 10
    n_action_steps: int = 4
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
        .stream-container { position: relative; cursor: crosshair; }
        .stream { border: 2px solid #e94560; border-radius: 8px; display: block; }
        .controls { margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
        button { padding: 8px 20px; font-size: 14px; font-family: monospace; cursor: pointer;
                 background: #16213e; color: #eee; border: 1px solid #e94560; border-radius: 4px; }
        button:hover { background: #e94560; }
        .active { background: #e94560 !important; }
        .sliders { margin-top: 12px; display: flex; gap: 30px; }
        .slider-group { display: flex; flex-direction: column; align-items: center; }
        .slider-group label { font-size: 12px; color: #aaa; margin-bottom: 4px; }
        .slider-group input { width: 150px; }
        .slider-group .val { font-size: 14px; color: #e94560; margin-top: 2px; }
        .info { margin-top: 10px; font-size: 14px; color: #aaa; }
        #stats { margin-top: 5px; font-size: 13px; }
        #mode-label { margin-top: 5px; font-size: 13px; color: #e94560; }
    </style>
</head>
<body>
    <h1>LOBE — PushT Policy Eval</h1>
    <div class="stream-container" id="streamContainer">
        <img class="stream" src="/stream" width="{{ size }}" height="{{ size }}" id="streamImg" />
    </div>
    <div class="controls">
        <button onclick="fetch('/api/reset')">Reset (R)</button>
        <button onclick="fetch('/api/pause')">Pause (Space)</button>
        <button id="modeBtn" onclick="toggleMode()">Mode: Watch (M)</button>
        <button onclick="fetch('/api/save_recording')">Save Recording</button>
    </div>
    <div class="sliders">
        <div class="slider-group">
            <label>Action Steps (execution horizon)</label>
            <input type="range" min="1" max="16" value="{{ n_action_steps }}" id="actSlider"
                   oninput="document.getElementById('actVal').textContent=this.value;
                            fetch('/api/set?n_action_steps='+this.value)">
            <div class="val" id="actVal">{{ n_action_steps }}</div>
        </div>
        <div class="slider-group">
            <label>Inference Steps (ODE solver)</label>
            <input type="range" min="1" max="50" value="{{ inference_steps }}" id="infSlider"
                   oninput="document.getElementById('infVal').textContent=this.value;
                            fetch('/api/set?num_inference_steps='+this.value)">
            <div class="val" id="infVal">{{ inference_steps }}</div>
        </div>
    </div>
    <div class="info">{{ policy_type }} | {{ checkpoint }}</div>
    <div id="mode-label"></div>
    <div id="stats"></div>
    <script>
        let mode = 'watch';
        let mouseDown = false;
        const img = document.getElementById('streamImg');
        const envSize = 512;

        function toggleMode() {
            mode = mode === 'watch' ? 'intervene' : 'watch';
            fetch('/api/set?mode=' + mode);
            const btn = document.getElementById('modeBtn');
            btn.textContent = 'Mode: ' + mode.charAt(0).toUpperCase() + mode.slice(1) + ' (M)';
            btn.classList.toggle('active', mode === 'intervene');
        }

        function sendMouse(e) {
            const rect = img.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * envSize;
            const y = (e.clientY - rect.top) / rect.height * envSize;
            fetch(`/api/mouse?x=${x.toFixed(1)}&y=${y.toFixed(1)}&active=true`);
        }

        // Prevent image drag
        img.addEventListener('dragstart', e => e.preventDefault());
        // In intervene mode: just move mouse over image to control (no click needed)
        img.addEventListener('mousemove', e => { if (mode === 'intervene') sendMouse(e); });
        img.addEventListener('mouseleave', () => { fetch('/api/mouse?active=false'); });

        setInterval(async () => {
            const r = await fetch('/api/stats');
            const d = await r.json();
            document.getElementById('stats').innerText =
                `Episode ${d.episode} | Step ${d.step} | ` +
                `Coverage ${(d.coverage*100).toFixed(1)}% | Max ${(d.max_coverage*100).toFixed(1)}% | ` +
                `Success ${d.total_successes}/${d.total_episodes} | ${d.fps.toFixed(1)} FPS`;
            document.getElementById('mode-label').innerText =
                d.mouse_active ? 'MOUSE CONTROL ACTIVE' : '';
        }, 500);

        document.addEventListener('keydown', e => {
            if (e.key === 'r') fetch('/api/reset');
            if (e.key === ' ') { e.preventDefault(); fetch('/api/pause'); }
            if (e.key === 'm') toggleMode();
        });
    </script>
</body>
</html>
"""


def draw_overlay(frame, info):
    """Draw HUD on frame."""
    overlay = frame.copy()
    cov = info.get("coverage", 0)
    max_cov = info.get("max_coverage", 0)
    color = (0, 255, 0) if cov > 0.95 else (0, 255, 255) if cov > 0.8 else (255, 255, 255)
    texts = [
        f"Ep {info.get('episode', 0)} | Step {info.get('step', 0)}/{info.get('max_steps', 300)}",
        f"Coverage: {cov:.1%} | Max: {max_cov:.1%} | Act: {info.get('n_action_steps', 4)}",
        f"Inf steps: {info.get('num_inference_steps', 10)} | Latency: {info.get('latency_ms', 0):.1f}ms",
    ]
    for i, text in enumerate(texts):
        cv2.putText(overlay, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def policy_loop(args: Args):
    """Background thread: runs policy rollouts and updates shared state."""
    logger.info("Loading policy...")
    dataset, features = pusht.load_dataset(args.dataset_repo_id)
    stats = dataset.meta.stats

    def build_policy():
        with state_lock:
            n_act = state["n_action_steps"]
            n_inf = state["num_inference_steps"]
        p = pusht.create_policy(
            args.policy_type,
            features,
            stats,
            n_action_steps=n_act,
            num_inference_steps=n_inf,
        )
        pusht.load_checkpoint(p, args.checkpoint, args.device)
        p.to(args.device)
        p.eval()
        logger.info(f"Policy built: act_steps={n_act}, inf_steps={n_inf}")
        return p

    policy = build_policy()
    logger.info(f"Policy loaded: {sum(p.numel() for p in policy.parameters()):,} params")

    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    episode = 0
    total_successes = 0

    while state["running"]:
        # Rebuild policy if params changed
        if state["rebuild_policy"]:
            with state_lock:
                state["rebuild_policy"] = False
            policy = build_policy()

        obs, _ = env.reset(seed=args.seed + episode)
        policy.reset()
        episode += 1
        max_coverage = 0.0

        with state_lock:
            state["reset"] = False

        step = 0
        while step < args.max_steps and state["running"]:
            if state["paused"]:
                time.sleep(0.05)
                continue
            if state["reset"] or state["rebuild_policy"]:
                break

            t0 = time.perf_counter()

            # Check if mouse is controlling
            with state_lock:
                mouse_active = state["mouse_active"] and state["mode"] == "intervene"
                mouse_pos = state["mouse_pos"]

            if mouse_active and mouse_pos is not None:
                action_np = mouse_pos.copy()
                policy.reset()  # reset action queue since we're overriding
                latency = 0.0
            else:
                batch = pusht.obs_to_batch(obs, args.device)
                with torch.no_grad():
                    action = policy.select_action(batch)
                action_np = action[0].cpu().numpy().clip(0, 512)
                latency = (time.perf_counter() - t0) * 1000
            obs, reward, terminated, truncated, info = env.step(action_np)
            step += 1
            coverage = info.get("coverage", 0)
            max_coverage = max(max_coverage, coverage)

            # Record action during mouse control
            if mouse_active:
                with state_lock:
                    state["current_recording"].append({"action": action_np.tolist(), "step": step})

            frame = env.render()
            frame = cv2.resize(frame, (args.render_size, args.render_size), interpolation=cv2.INTER_NEAREST)

            with state_lock:
                n_act = state["n_action_steps"]
                n_inf = state["num_inference_steps"]

            info_dict = {
                "episode": episode,
                "step": step,
                "max_steps": args.max_steps,
                "coverage": coverage,
                "max_coverage": max_coverage,
                "latency_ms": latency,
                "n_action_steps": n_act,
                "num_inference_steps": n_inf,
            }
            frame = draw_overlay(frame, info_dict)

            with state_lock:
                state["frame"] = frame
                state["info"] = {
                    "episode": episode,
                    "step": step,
                    "coverage": coverage,
                    "max_coverage": max_coverage,
                    "total_successes": total_successes,
                    "total_episodes": episode - 1,
                    "fps": 1000.0 / max(latency, 0.1),
                    "mouse_active": mouse_active,
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
        time.sleep(1.0 / 30)


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        size=app.config["render_size"],
        policy_type=app.config["policy_type"],
        checkpoint=app.config.get("checkpoint") or "random",
        inference_steps=state["num_inference_steps"],
        n_action_steps=state["n_action_steps"],
    )


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    with state_lock:
        return json.dumps(
            state.get(
                "info",
                {
                    "episode": 0,
                    "step": 0,
                    "coverage": 0,
                    "max_coverage": 0,
                    "total_successes": 0,
                    "total_episodes": 0,
                    "fps": 0,
                },
            )
        )


@app.route("/api/reset")
def api_reset():
    with state_lock:
        state["reset"] = True
    return "ok"


@app.route("/api/pause")
def api_pause():
    with state_lock:
        state["paused"] = not state["paused"]
    return "ok"


@app.route("/api/mouse")
def api_mouse():
    import numpy as np

    active = request.args.get("active", "false") == "true"
    with state_lock:
        state["mouse_active"] = active
        if "x" in request.args and "y" in request.args:
            state["mouse_pos"] = np.array([float(request.args["x"]), float(request.args["y"])])
    return "ok"


@app.route("/api/save_recording")
def api_save_recording():
    from pathlib import Path

    with state_lock:
        recording = state["current_recording"].copy()
        state["current_recording"] = []
    if not recording:
        return json.dumps({"saved": False, "reason": "empty"})
    save_dir = Path("recordings")
    save_dir.mkdir(exist_ok=True)
    path = save_dir / f"episode_{int(time.time())}.json"
    path.write_text(json.dumps(recording, indent=2))
    logger.info(f"Saved recording: {path} ({len(recording)} steps)")
    return json.dumps({"saved": True, "path": str(path), "steps": len(recording)})


@app.route("/api/set")
def api_set():
    changed = False
    if "n_action_steps" in request.args:
        val = int(request.args["n_action_steps"])
        with state_lock:
            if state["n_action_steps"] != val:
                state["n_action_steps"] = val
                state["rebuild_policy"] = True
                changed = True
    if "num_inference_steps" in request.args:
        val = int(request.args["num_inference_steps"])
        with state_lock:
            state["num_inference_steps"] = val
            state["rebuild_policy"] = True
            changed = True
    if "mode" in request.args:
        with state_lock:
            state["mode"] = request.args["mode"]
            changed = True
    return json.dumps({"changed": changed})


def main():
    args = tyro.cli(Args)

    app.config["render_size"] = args.render_size
    app.config["policy_type"] = args.policy_type
    app.config["checkpoint"] = args.checkpoint

    with state_lock:
        state["n_action_steps"] = args.n_action_steps
        state["num_inference_steps"] = args.num_inference_steps

    policy_thread = threading.Thread(target=policy_loop, args=(args,), daemon=True)
    policy_thread.start()

    logger.info(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
