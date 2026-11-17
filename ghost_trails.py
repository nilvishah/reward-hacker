import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # draw without opening a window

import argparse
from pathlib import Path

import gymnasium as gym
import pygame
from stable_baselines3 import PPO

from preview_style import W, H, glow, make_background, to_screen

COLORS = {"landed": (90, 255, 140), "crashed": (255, 80, 80), "timeout": (255, 170, 60)}


def collect(model_path: str, episodes: int):
    """Fly the agent on fixed seeds and record each flight path and outcome."""
    env = gym.make("LunarLander-v3")
    u = env.unwrapped
    model = PPO.load(model_path)
    flights, ground = [], None
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        if ground is None:
            pts = [(p[0][0], p[0][1]) for p in u.sky_polys] + [tuple(u.sky_polys[-1][1])]
            ground = (pts, (u.helipad_x1, u.helipad_x2, u.helipad_y))
        path = []
        while True:
            action = int(model.predict(obs, deterministic=True)[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            p = u.lander.position
            path.append(to_screen(p[0], p[1]))
            if terminated or truncated:
                outcome = "timeout" if truncated else ("landed" if reward > 0 else "crashed")
                break
        flights.append((path, outcome))
    env.close()
    return flights, ground


def draw_panel(surface, bg, flights, ground, title, sub, color, fonts):
    font, title_font, sub_font = fonts
    surface.blit(bg, (0, 0))

    # Ground and pad
    pts = [to_screen(x, y) for x, y in ground[0]]
    pygame.draw.polygon(surface, (38, 36, 52), pts + [(W, H), (0, H)])
    pygame.draw.lines(surface, (150, 140, 200), False, pts, 2)
    x1, x2, py = ground[1]
    a, b = to_screen(x1, py), to_screen(x2, py)
    glow(surface, ((a[0] + b[0]) // 2, a[1]), 80, (0, 220, 255))
    pygame.draw.line(surface, (0, 230, 255), a, b, 4)
    for fx in (x1, x2):
        base = to_screen(fx, py)
        top = (base[0], base[1] - 40)
        pygame.draw.line(surface, (220, 220, 230), base, top, 2)
        pygame.draw.polygon(surface, (255, 200, 40), [top, (top[0] + 16, top[1] + 6), (top[0], top[1] + 12)])

    # Every flight path, semi-transparent so overlapping paths glow brighter
    for path, outcome in flights:
        if len(path) < 2:
            continue
        layer = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.lines(layer, (*COLORS[outcome], 110), False, path, 2)
        surface.blit(layer, (0, 0))
        pygame.draw.circle(surface, COLORS[outcome], path[-1], 4)

    # Title box
    t = title_font.render(title, True, color)
    s = sub_font.render(sub, True, (200, 200, 220))
    box = pygame.Rect(0, 0, max(t.get_width(), s.get_width()) + 40, 70)
    box.midtop = (W // 2, 16)
    pygame.draw.rect(surface, (10, 12, 26), box, border_radius=10)
    pygame.draw.rect(surface, color, box, 2, border_radius=10)
    surface.blit(t, t.get_rect(midtop=(box.centerx, box.top + 8)))
    surface.blit(s, s.get_rect(midtop=(box.centerx, box.top + 42)))

    # Stats line
    n = len(flights)
    counts = {k: sum(1 for _, o in flights if o == k) for k in COLORS}
    parts = [
        (f"{counts['landed']} landed", COLORS["landed"]),
        (f"{counts['crashed']} crashed", COLORS["crashed"]),
        (f"{counts['timeout']} never landed", COLORS["timeout"]),
    ]
    rendered = [font.render(f"{n} attempts  ·  ", True, (220, 220, 235))]
    for i, (text, c) in enumerate(parts):
        rendered.append(font.render(text, True, c))
        if i < len(parts) - 1:
            rendered.append(font.render("  ·  ", True, (220, 220, 235)))
    total = sum(r.get_width() for r in rendered)
    x, y = W // 2 - total // 2, H - 40
    bar = pygame.Rect(x - 16, y - 8, total + 32, 34)
    pygame.draw.rect(surface, (10, 12, 26), bar, border_radius=8)
    for r in rendered:
        surface.blit(r, (x, y))
        x += r.get_width()


def main():
    parser = argparse.ArgumentParser(description="Draw every flight path of two agents side by side.")
    parser.add_argument("--left", default="ppo_lunar_baseline")
    parser.add_argument("--right", default="models/ppo_hover_hack_seed0")
    parser.add_argument("--left-title", default="NORMAL REWARD")
    parser.add_argument("--right-title", default="HACKED REWARD")
    parser.add_argument("--left-sub", default="the game's built-in points")
    parser.add_argument("--right-sub", default="+ bonus for staying upright and slow")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", default="results/ghost_trails.png")
    args = parser.parse_args()

    pygame.init()
    fonts = (
        pygame.font.SysFont("menlo", 18),
        pygame.font.SysFont("menlo", 26, bold=True),
        pygame.font.SysFont("menlo", 15),
    )
    bg = make_background()

    print("Flying left agent...")
    left_flights, ground = collect(args.left, args.episodes)
    print("Flying right agent...")
    right_flights, _ = collect(args.right, args.episodes)

    left, right = pygame.Surface((W, H)), pygame.Surface((W, H))
    draw_panel(left, bg, left_flights, ground, args.left_title, args.left_sub, (90, 255, 140), fonts)
    draw_panel(right, bg, right_flights, ground, args.right_title, args.right_sub, (255, 140, 60), fonts)

    canvas = pygame.Surface((W * 2, H))
    canvas.blit(left, (0, 0))
    canvas.blit(right, (W, 0))
    pygame.draw.line(canvas, (90, 90, 130), (W, 0), (W, H), 3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(canvas, args.out)
    pygame.quit()
    print(f"Saved image to {args.out}")


if __name__ == "__main__":
    main()
