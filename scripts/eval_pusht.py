"""PushT Interactive Eval — pygame viewer for trained policies.

Modes:
  watch   — policy runs autonomously, you observe
  intervene — policy runs, click+drag to take over mid-rollout
  record  — push T yourself with mouse, save episodes

Keyboard controls (during rollout):
  SPACE     Toggle pause
  M         Toggle mouse intervention (watch <-> intervene)
  R         Reset episode
  S         Save current episode video
  UP/DOWN   Change inference steps (+/- 1)
  LEFT/RIGHT Change action chunk size (+/- 1)
  Q/ESC     Quit
  1-9       Set inference steps directly

Usage:
    uv run python scripts/eval_pusht.py --mode watch
    uv run python scripts/eval_pusht.py --mode intervene --checkpoint checkpoints/fm-pusht
    uv run python scripts/eval_pusht.py --mode record --save_dir recordings/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import pygame
import torch
import tyro
from loguru import logger

import lobe.video_compat  # noqa: F401
from lobe import pusht

WINDOW_SIZE = 512
FPS = int(pusht.FPS)


@dataclass
class Args:
    mode: str = "watch"  # watch | intervene | record
    policy_type: str = "flow_matching"  # flow_matching | diffusion
    checkpoint: str = ""
    num_inference_steps: int = 10
    horizon: int = pusht.HORIZON
    n_action_steps: int = pusht.N_ACTION_STEPS
    max_episodes: int = 0  # 0 = unlimited
    max_steps: int = pusht.MAX_STEPS
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "recordings/"
    save_video: bool = True
    dataset_repo_id: str = pusht.DEFAULT_DATASET


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def make_policy(args: Args):
    dataset, features = pusht.load_dataset(args.dataset_repo_id)
    policy = pusht.create_policy(
        args.policy_type,
        features,
        dataset.meta.stats,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
    )
    pusht.load_checkpoint(policy, args.checkpoint, args.device)
    return policy


def obs_to_policy_batch(obs: dict, device: str) -> dict:
    """Convert a single gym obs dict to policy input batch."""
    return pusht.obs_to_batch(obs, device)


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------


def draw_hud(
    surface: pygame.Surface,
    info: dict,
    mode: str,
    paused: bool,
    intervention: bool,
    n_inf_steps: int,
    n_act_steps: int,
    episode: int,
    step: int,
    fps_actual: float,
):
    font = pygame.font.SysFont("monospace", 14, bold=True)
    lines = [
        f"Mode: {mode.upper()}{'  [PAUSED]' if paused else ''}",
        f"Episode: {episode}  Step: {step}",
        f"Inference Steps: {n_inf_steps}  Action Chunk: {n_act_steps}",
        f"FPS: {fps_actual:.1f}",
        f"Reward: {info.get('coverage', 0):.3f}  {'SUCCESS' if info.get('is_success') else ''}",
    ]
    if intervention:
        lines.append("MOUSE INTERVENTION ACTIVE")

    y = 5
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        bg = pygame.Surface((text.get_width() + 6, text.get_height() + 2))
        bg.set_alpha(160)
        bg.fill((0, 0, 0))
        surface.blit(bg, (3, y - 1))
        surface.blit(text, (6, y))
        y += 18


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    args = tyro.cli(Args)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"Device: {device}")
    args.device = device

    # Load policy (not needed in record mode but load anyway for switching)
    policy = None
    if args.mode != "record":
        logger.info("Loading policy...")
        policy = make_policy(args)
        policy.to(device)
        policy.eval()
        n_params = sum(p.numel() for p in policy.parameters())
        logger.info(f"Policy: {args.policy_type}, {n_params:,} params")

    # Init pygame
    pygame.init()
    pygame.display.set_caption("LOBE — PushT Eval")
    window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    pygame.font.init()

    # Create env
    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")

    mode = args.mode
    n_inf_steps = args.num_inference_steps
    n_act_steps = args.n_action_steps
    paused = False
    running = True
    episode_num = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    while running:
        episode_num += 1
        if args.max_episodes > 0 and episode_num > args.max_episodes:
            break

        obs, info = env.reset(seed=args.seed + episode_num - 1)
        obs_history = [obs, obs]  # pad to n_obs_steps=2
        frames = []
        rewards = []
        step = 0
        done = False
        intervention_active = False
        fps_actual = FPS

        if policy is not None:
            policy.reset()
            # Update inference steps if changed via keyboard
            if hasattr(policy, "flow_matching") and hasattr(policy.flow_matching, "num_inference_steps"):
                policy.flow_matching.num_inference_steps = n_inf_steps

        logger.info(f"Episode {episode_num} | mode={mode} inf_steps={n_inf_steps} act_steps={n_act_steps}")

        while not done and running:
            t0 = time.perf_counter()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_m:
                        if mode == "watch":
                            mode = "intervene"
                        elif mode == "intervene":
                            mode = "watch"
                        logger.info(f"Mode switched to: {mode}")
                    elif event.key == pygame.K_r:
                        done = True  # trigger reset
                    elif event.key == pygame.K_s:
                        if frames:
                            video_path = save_dir / f"episode_{episode_num:04d}.mp4"
                            save_video(frames, video_path)
                            logger.info(f"Saved video: {video_path}")
                    elif event.key == pygame.K_UP:
                        n_inf_steps = min(n_inf_steps + 1, 100)
                        if policy and hasattr(policy, "flow_matching"):
                            policy.flow_matching.num_inference_steps = n_inf_steps
                            policy.reset()
                        logger.info(f"Inference steps: {n_inf_steps}")
                    elif event.key == pygame.K_DOWN:
                        n_inf_steps = max(n_inf_steps - 1, 1)
                        if policy and hasattr(policy, "flow_matching"):
                            policy.flow_matching.num_inference_steps = n_inf_steps
                            policy.reset()
                        logger.info(f"Inference steps: {n_inf_steps}")
                    elif event.key in range(pygame.K_1, pygame.K_9 + 1):
                        n_inf_steps = event.key - pygame.K_0
                        if policy and hasattr(policy, "flow_matching"):
                            policy.flow_matching.num_inference_steps = n_inf_steps
                            policy.reset()
                        logger.info(f"Inference steps: {n_inf_steps}")
                    elif event.key == pygame.K_RIGHT:
                        n_act_steps = min(n_act_steps + 1, 32)
                        logger.info(f"Action steps: {n_act_steps}")
                    elif event.key == pygame.K_LEFT:
                        n_act_steps = max(n_act_steps - 1, 1)
                        logger.info(f"Action steps: {n_act_steps}")

            if paused:
                # Still render current frame
                frame = env.render()
                surf = pygame.surfarray.make_surface(
                    np.transpose(cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE)), (1, 0, 2))
                )
                window.blit(surf, (0, 0))
                draw_hud(
                    window,
                    info,
                    mode,
                    paused,
                    intervention_active,
                    n_inf_steps,
                    n_act_steps,
                    episode_num,
                    step,
                    fps_actual,
                )
                pygame.display.flip()
                clock.tick(30)
                continue

            step += 1
            if step > args.max_steps:
                done = True
                continue

            # Determine action
            action = None
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]

            if mode == "record":
                # Mouse controls agent directly
                if mouse_pressed:
                    # Map window coords to env action space [0, 512]
                    action = np.array([mouse_pos[0], mouse_pos[1]], dtype=np.float32)
                    intervention_active = True
                else:
                    # Stay in place
                    action = np.array(obs["agent_pos"], dtype=np.float32)
                    intervention_active = False

            elif mode == "intervene":
                # Check if mouse is near agent and pressed
                agent_screen_pos = obs["agent_pos"] / 512 * WINDOW_SIZE
                dist = np.linalg.norm(np.array(mouse_pos) - agent_screen_pos)
                if mouse_pressed and dist < 40:
                    intervention_active = True
                if not mouse_pressed:
                    intervention_active = False

                if intervention_active:
                    action = np.array(
                        [mouse_pos[0] / WINDOW_SIZE * 512, mouse_pos[1] / WINDOW_SIZE * 512], dtype=np.float32
                    )
                    if policy:
                        policy.reset()  # clear action cache on intervention

            if action is None and policy is not None:
                # Policy generates action
                intervention_active = False
                batch = obs_to_policy_batch(obs, device)
                with torch.no_grad():
                    policy_action = policy.select_action(batch)
                action = policy_action[0].cpu().numpy()

            if action is None:
                # Fallback — stay in place
                action = np.array(obs["agent_pos"], dtype=np.float32)

            # Step env
            action = np.clip(action, 0, 512)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            rewards.append(reward)

            # Render
            frame = env.render()
            frames.append(frame)

            # Scale to window and display
            display_frame = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
            surf = pygame.surfarray.make_surface(np.transpose(display_frame, (1, 0, 2)))
            window.blit(surf, (0, 0))
            draw_hud(
                window,
                info,
                mode,
                paused,
                intervention_active,
                n_inf_steps,
                n_act_steps,
                episode_num,
                step,
                fps_actual,
            )
            pygame.display.flip()

            if terminated or truncated:
                done = True

            dt = time.perf_counter() - t0
            fps_actual = 1.0 / max(dt, 1e-6)
            clock.tick(FPS)

        # Episode done
        avg_reward = np.mean(rewards) if rewards else 0
        max_reward = np.max(rewards) if rewards else 0
        logger.info(
            f"Episode {episode_num} done: {step} steps, "
            f"avg_reward={avg_reward:.4f}, max_reward={max_reward:.4f}, "
            f"success={info.get('is_success', False)}"
        )

        # Auto-save video
        if args.save_video and frames:
            video_path = save_dir / f"episode_{episode_num:04d}.mp4"
            save_video(frames, video_path)
            logger.info(f"Saved: {video_path}")

    env.close()
    pygame.quit()
    logger.info("Done.")


def save_video(frames: list[np.ndarray], path: Path, fps: int = FPS):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    main()
