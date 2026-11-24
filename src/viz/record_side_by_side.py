import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # draw without opening a window

import argparse
import math
from pathlib import Path

import gymnasium as gym
import imageio
import pygame
from stable_baselines3 import PPO

from src.viz.preview_style import W, H, draw_scene, make_background, to_screen


class Runner:
    """Plays one trained rocket and keeps the state needed for drawing."""

    def __init__(self, model_path: str, seed: int):
        self.env = gym.make("LunarLander-v3")
        self.u = self.env.unwrapped
        self.model = PPO.load(model_path)
        self.obs, _ = self.env.reset(seed=seed)
        self.trail, self.score, self.frame = [], 0.0, 0
        self.action, self.stamp, self.done = 0, None, False

    def step(self):
        if self.done:
            return
        self.action = int(self.model.predict(self.obs, deterministic=True)[0])
        self.obs, reward, terminated, truncated, _ = self.env.step(self.action)
        self.score += reward
        self.frame += 1
        p = self.u.lander.position
        self.trail = (self.trail + [to_screen(p[0], p[1])])[-300:]
        if terminated or truncated:
            self.done, self.action = True, 0
            if truncated:
                self.stamp = ("NEVER LANDED", (255, 80, 80))
            elif reward > 0:
                self.stamp = ("TOUCHDOWN", (90, 255, 140))
            else:
                self.stamp = ("CRASHED", (255, 80, 80))

    def hud(self):
        o = self.obs
        return [
            f"T+ {self.frame:04d}",
            f"ALT    {o[1]:6.2f}",
            f"V-SPD  {o[3]:6.2f}",
            f"H-SPD  {o[2]:6.2f}",
            f"TILT   {math.degrees(o[4]):6.1f}°",
            f"SCORE  {self.score:6.1f}",
        ]


def draw_label(surface, title, subtitle, color, title_font, sub_font):
    t = title_font.render(title, True, color)
    s = sub_font.render(subtitle, True, (200, 200, 220))
    box = pygame.Rect(0, 0, max(t.get_width(), s.get_width()) + 40, 70)
    box.midtop = (W // 2 + 100, 16)
    pygame.draw.rect(surface, (10, 12, 26), box, border_radius=10)
    pygame.draw.rect(surface, color, box, 2, border_radius=10)
    surface.blit(t, t.get_rect(midtop=(box.centerx, box.top + 8)))
    surface.blit(s, s.get_rect(midtop=(box.centerx, box.top + 42)))


def main():
    parser = argparse.ArgumentParser(description="Record two agents side by side.")
    parser.add_argument("--left", default="ppo_lunar_baseline")
    parser.add_argument("--right", default="models/ppo_hover_hack_seed0")
    parser.add_argument("--left-title", default="NORMAL REWARD")
    parser.add_argument("--right-title", default="HACKED REWARD")
    parser.add_argument("--left-sub", default="the game's built-in points")
    parser.add_argument("--right-sub", default="+ bonus for staying upright and slow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/videos/side_by_side.mp4")
    args = parser.parse_args()

    pygame.init()
    font = pygame.font.SysFont("menlo", 18)
    big = pygame.font.SysFont("menlo", 56, bold=True)
    title_font = pygame.font.SysFont("menlo", 26, bold=True)
    sub_font = pygame.font.SysFont("menlo", 15)
    bg = make_background()

    left, right = Runner(args.left, args.seed), Runner(args.right, args.seed)
    left_surf, right_surf = pygame.Surface((W, H)), pygame.Surface((W, H))
    canvas = pygame.Surface((W * 2, H))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=50, codec="libx264", quality=8, macro_block_size=8)

    hold = 0
    while hold < 100:  # keep both results on screen for 2 seconds at the end
        left.step()
        right.step()
        if left.done and right.done:
            hold += 1

        for runner, surf, title, sub, color in [
            (left, left_surf, args.left_title, args.left_sub, (90, 255, 140)),
            (right, right_surf, args.right_title, args.right_sub, (255, 140, 60)),
        ]:
            draw_scene(surf, bg, runner.u, runner.trail, runner.action, runner.hud(), runner.stamp, font, big)
            draw_label(surf, title, sub, color, title_font, sub_font)

        canvas.blit(left_surf, (0, 0))
        canvas.blit(right_surf, (W, 0))
        pygame.draw.line(canvas, (90, 90, 130), (W, 0), (W, H), 3)
        writer.append_data(pygame.surfarray.array3d(canvas).swapaxes(0, 1))

    writer.close()
    left.env.close()
    right.env.close()
    pygame.quit()
    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
