import math
import random
import sys

import gymnasium as gym
import pygame
from stable_baselines3 import PPO

S = 45  # pixels per world unit
W, H = 900, 600
BG_TOP, BG_BOTTOM = (6, 8, 22), (22, 18, 48)
LANDER = [(-14, 17), (-17, 0), (-17, -10), (17, -10), (17, 0), (14, 17)]
LANDER = [(px / 30, py / 30) for px, py in LANDER]


def to_screen(x, y):
    return int(x * S), int(H - y * S)


def local_to_screen(cx, cy, angle, lx, ly):
    wx = cx + lx * math.cos(angle) - ly * math.sin(angle)
    wy = cy + lx * math.sin(angle) + ly * math.cos(angle)
    return to_screen(wx, wy)


def make_background():
    bg = pygame.Surface((W, H))
    for row in range(H):
        t = row / H
        color = [int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)]
        pygame.draw.line(bg, color, (0, row), (W, row))
    rng = random.Random(7)
    for _ in range(160):
        b = rng.randint(90, 255)
        bg.set_at((rng.randrange(W), rng.randrange(int(H * 0.8))), (b, b, b))
    return bg


def glow(surface, pos, radius, color):
    g = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for r in range(radius, 0, -3):
        pygame.draw.circle(g, (*color, int(40 * (1 - r / radius))), (radius, radius), r)
    surface.blit(g, (pos[0] - radius, pos[1] - radius))


def draw_scene(screen, bg, u, trail, action, hud, stamp, font, big):
    screen.blit(bg, (0, 0))

    # Moon surface
    pts = [to_screen(p[0][0], p[0][1]) for p in u.sky_polys]
    last = u.sky_polys[-1][1]
    pts.append(to_screen(last[0], last[1]))
    pygame.draw.polygon(screen, (38, 36, 52), pts + [(W, H), (0, H)])
    pygame.draw.lines(screen, (150, 140, 200), False, pts, 2)

    # Landing pad and flags
    x1, x2, py = u.helipad_x1, u.helipad_x2, u.helipad_y
    a, b = to_screen(x1, py), to_screen(x2, py)
    glow(screen, ((a[0] + b[0]) // 2, a[1]), 80, (0, 220, 255))
    pygame.draw.line(screen, (0, 230, 255), a, b, 4)
    for fx in (x1, x2):
        base = to_screen(fx, py)
        top = (base[0], base[1] - 40)
        pygame.draw.line(screen, (220, 220, 230), base, top, 2)
        pygame.draw.polygon(screen, (255, 200, 40), [top, (top[0] + 16, top[1] + 6), (top[0], top[1] + 12)])

    # Flight trail
    for i in range(1, len(trail)):
        t = i / len(trail)
        pygame.draw.line(screen, (int(40 + 200 * t), int(80 + 120 * t), 255), trail[i - 1], trail[i], 2)

    # Engine flames
    cx, cy = u.lander.position[0], u.lander.position[1]
    ang = u.lander.angle
    if action == 2:
        length = random.uniform(0.7, 1.2)
        outer = [local_to_screen(cx, cy, ang, *p) for p in [(-0.25, -0.33), (0.25, -0.33), (0, -0.33 - length)]]
        inner = [local_to_screen(cx, cy, ang, *p) for p in [(-0.12, -0.33), (0.12, -0.33), (0, -0.33 - length * 0.6)]]
        glow(screen, outer[2], 40, (255, 140, 40))
        pygame.draw.polygon(screen, (255, 120, 30), outer)
        pygame.draw.polygon(screen, (255, 230, 120), inner)
    elif action in (1, 3):
        side = 1 if action == 1 else -1
        base = local_to_screen(cx, cy, ang, side * 0.57, 0.35)
        tip = local_to_screen(cx, cy, ang, side * 0.95, 0.35)
        pygame.draw.line(screen, (255, 210, 120), base, tip, 5)

    # Legs (green when touching the ground)
    for leg in u.legs:
        lx, ly, la = leg.position[0], leg.position[1], leg.angle
        hip = local_to_screen(lx, ly, la, 0, 0.27)
        foot = local_to_screen(lx, ly, la, 0, -0.27)
        color = (120, 255, 160) if leg.ground_contact else (200, 200, 215)
        pygame.draw.line(screen, color, hip, foot, 3)
        pygame.draw.circle(screen, color, foot, 3)

    # Rocket body
    body = [local_to_screen(cx, cy, ang, *p) for p in LANDER]
    pygame.draw.polygon(screen, (235, 235, 245), body)
    pygame.draw.polygon(screen, (120, 120, 150), body, 2)
    pygame.draw.circle(screen, (0, 200, 255), local_to_screen(cx, cy, ang, 0, 0.25), 6)

    # Mission-control panel
    panel = pygame.Surface((230, 170), pygame.SRCALPHA)
    panel.fill((10, 14, 30, 180))
    screen.blit(panel, (16, 16))
    for i, line in enumerate(hud):
        screen.blit(font.render(line, True, (140, 230, 255)), (28, 26 + i * 24))

    # Result stamp
    if stamp:
        text, color = stamp
        surf = big.render(text, True, color)
        rect = surf.get_rect(center=(W // 2, H // 3))
        box = rect.inflate(40, 20)
        pygame.draw.rect(screen, (10, 10, 20), box, border_radius=12)
        pygame.draw.rect(screen, color, box, 3, border_radius=12)
        screen.blit(surf, rect)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "ppo_lunar_baseline"
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"RewardHacker preview: {model_path}")
    font = pygame.font.SysFont("menlo", 18)
    big = pygame.font.SysFont("menlo", 56, bold=True)
    bg = make_background()
    clock = pygame.time.Clock()

    env = gym.make("LunarLander-v3")
    u = env.unwrapped
    model = PPO.load(model_path)

    for episode in range(3):
        obs, _ = env.reset(seed=episode)
        trail, score, fuel, frame, action, stamp = [], 0.0, 0.0, 0, 0, None
        done, hold = False, 0
        while hold < 100:  # keep the result on screen for 2 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            if not done:
                action = int(model.predict(obs, deterministic=True)[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                score += reward
                frame += 1
                fuel += 0.3 if action == 2 else (0.03 if action in (1, 3) else 0.0)
                p = u.lander.position
                trail = (trail + [to_screen(p[0], p[1])])[-300:]
                if terminated or truncated:
                    done, action = True, 0
                    if truncated:
                        stamp = ("NEVER LANDED", (255, 80, 80))
                    elif reward > 0:
                        stamp = ("TOUCHDOWN", (90, 255, 140))
                    else:
                        stamp = ("CRASHED", (255, 80, 80))
            else:
                hold += 1

            hud = [
                f"T+ {frame:04d}",
                f"ALT    {obs[1]:6.2f}",
                f"V-SPD  {obs[3]:6.2f}",
                f"H-SPD  {obs[2]:6.2f}",
                f"TILT   {math.degrees(obs[4]):6.1f}°",
                f"SCORE  {score:6.1f}",
            ]
            draw_scene(screen, bg, u, trail, action, hud, stamp, font, big)
            pygame.display.flip()
            clock.tick(50)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
