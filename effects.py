"""
effects.py
----------
A lightweight, dependency-free particle animation engine built entirely on
OpenCV drawing primitives (circles, lines, ellipses, polygons + alpha
blending for glow). No external image assets are required.

Structure
---------
- Particle: a single moving point (position, velocity, life, color, size).
- ParticleEffect: base class. Subclasses implement `spawn()` (what particles
  to create when the effect triggers) and may override `draw()` for custom
  shapes (hearts, stars, lightning bolts, etc).
- One subclass per gesture (10 total).
- EffectsManager: keeps track of currently-playing effects, enforces a
  per-gesture cooldown so the same effect can't spam-retrigger, and calls
  update()/draw() every frame from app.py.

All effects take `landmarks_px` (a list of 21 (x, y) pixel tuples, see
utils.landmarks_to_pixel_coords) so they can anchor themselves to specific
parts of the hand (fingertip, palm center, etc).
"""

import math
import random
import time

import cv2
import numpy as np

from utils import LM_INDEX_TIP


def _palm_center(landmarks_px):
    """Average of wrist + finger MCP-ish points, used as a stable anchor."""
    idxs = [0, 1, 5, 9, 13, 17]
    xs = [landmarks_px[i][0] for i in idxs]
    ys = [landmarks_px[i][1] for i in idxs]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def _clamp255(v):
    return max(0, min(255, int(v)))


# ---------------------------------------------------------------------------
# Core particle primitives
# ---------------------------------------------------------------------------
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "ax", "ay", "life", "max_life",
                 "size", "color", "extra")

    def __init__(self, x, y, vx=0.0, vy=0.0, ax=0.0, ay=0.0,
                 life=1.0, size=6, color=(255, 255, 255), extra=None):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.ax, self.ay = ax, ay
        self.life = life          # seconds remaining
        self.max_life = life
        self.size = size
        self.color = color        # BGR
        self.extra = extra or {}  # per-effect custom data

    def update(self, dt):
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    @property
    def alive(self):
        return self.life > 0

    @property
    def life_ratio(self):
        """1.0 = just born, 0.0 = about to die."""
        return max(0.0, self.life / self.max_life) if self.max_life > 0 else 0.0


class ParticleEffect:
    """Base class for every gesture effect."""

    duration = 1.5  # seconds the effect plays for after being triggered

    def __init__(self):
        self.particles = []
        self.start_time = time.time()

    def spawn(self, landmarks_px, frame_w, frame_h):
        """Override: populate self.particles based on hand position."""
        raise NotImplementedError

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, frame):
        """Default: draw every particle as a fading filled circle."""
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha + 0) for c in p.color)
            cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.size)),
                       color, -1, lineType=cv2.LINE_AA)

    def is_alive(self):
        return (time.time() - self.start_time) < self.duration and \
               (len(self.particles) > 0 or (time.time() - self.start_time) < 0.3)


# ---------------------------------------------------------------------------
# 1. Peace -> Floating balloons
# ---------------------------------------------------------------------------
class BalloonEffect(ParticleEffect):
    duration = 2.5
    COLORS = [(80, 60, 230), (60, 200, 255), (200, 120, 60), (120, 220, 120), (220, 90, 200)]

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        for i in range(6):
            x = cx + random.randint(-60, 60)
            y = cy + random.randint(-20, 20)
            self.particles.append(Particle(
                x, y,
                vx=random.uniform(-15, 15), vy=random.uniform(-90, -60),
                ax=0, ay=-5,
                life=self.duration, size=random.randint(18, 26),
                color=random.choice(self.COLORS),
                extra={"phase": random.uniform(0, math.tau), "sway": random.uniform(20, 40)},
            ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            sway = p.extra["sway"] * math.sin(p.extra["phase"] + p.life * 3)
            bx, by = int(p.x + sway), int(p.y)
            color = tuple(_clamp255(c * (0.4 + 0.6 * alpha)) for c in p.color)
            # balloon body
            cv2.ellipse(frame, (bx, by), (int(p.size), int(p.size * 1.25)),
                        0, 0, 360, color, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(frame, (bx - p.size // 3, by - p.size // 2),
                        (max(2, p.size // 4), max(2, p.size // 5)), -30, 0, 360,
                        tuple(_clamp255(c * 1.3) for c in color), -1, lineType=cv2.LINE_AA)
            # string
            cv2.line(frame, (bx, by + int(p.size * 1.2)),
                     (bx, by + int(p.size * 1.2) + 18), (180, 180, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 2. Heart Hands -> Falling hearts
# ---------------------------------------------------------------------------
class HeartsEffect(ParticleEffect):
    duration = 2.0

    @staticmethod
    def _heart_points(cx, cy, scale):
        pts = []
        for t_deg in range(0, 360, 12):
            t = math.radians(t_deg)
            hx = 16 * math.sin(t) ** 3
            hy = -(13 * math.cos(t) - 5 * math.cos(2 * t)
                   - 2 * math.cos(3 * t) - math.cos(4 * t))
            pts.append((int(cx + hx * scale), int(cy + hy * scale)))
        return np.array(pts, dtype=np.int32)

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        for i in range(10):
            x = cx + random.randint(-70, 70)
            y = cy + random.randint(-40, 10)
            self.particles.append(Particle(
                x, y, vx=random.uniform(-8, 8), vy=random.uniform(20, 45),
                ay=10, life=random.uniform(1.2, self.duration),
                size=random.uniform(0.6, 1.3),
                color=(120, 40, 220),
            ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha) for c in p.color)
            pts = self._heart_points(p.x, p.y, p.size)
            cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 3. Thumbs Up -> Confetti burst
# ---------------------------------------------------------------------------
class ConfettiEffect(ParticleEffect):
    duration = 1.6
    COLORS = [(60, 60, 255), (60, 255, 60), (255, 200, 40), (255, 60, 200), (60, 220, 255)]

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        for i in range(40):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(80, 260)
            self.particles.append(Particle(
                cx, cy,
                vx=math.cos(angle) * speed, vy=math.sin(angle) * speed - 120,
                ay=220, life=random.uniform(0.9, self.duration),
                size=random.randint(3, 6),
                color=random.choice(self.COLORS),
                extra={"angle": random.uniform(0, 360), "spin": random.uniform(-400, 400)},
            ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha) for c in p.color)
            p.extra["angle"] += p.extra["spin"] * 0.016
            rect = ((p.x, p.y), (p.size * 2, p.size), p.extra["angle"])
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(frame, [box], color, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 4. Open Palm -> Sparkles around the hand
# ---------------------------------------------------------------------------
class SparklesEffect(ParticleEffect):
    duration = 1.2

    def spawn(self, landmarks_px, frame_w, frame_h):
        # scatter sparkles around the outline of the hand (all 21 points)
        for (lx, ly) in landmarks_px:
            for _ in range(2):
                self.particles.append(Particle(
                    lx + random.randint(-18, 18), ly + random.randint(-18, 18),
                    vy=random.uniform(-15, -5),
                    life=random.uniform(0.4, self.duration),
                    size=random.uniform(2, 5),
                    color=(255, 255, 255),
                ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            s = p.size * (0.5 + 0.5 * alpha)
            color = (int(255 * alpha), int(255 * alpha), int(180 * alpha + 75))
            x, y = int(p.x), int(p.y)
            # 4-point sparkle (plus sign with tapered ends)
            cv2.line(frame, (x - int(s * 2), y), (x + int(s * 2), y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - int(s * 2)), (x, y + int(s * 2)), color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(1, int(s * 0.6)), color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 5. Wave -> Rainbow trail (follows index fingertip motion history)
# ---------------------------------------------------------------------------
class RainbowTrailEffect(ParticleEffect):
    duration = 1.4
    RAINBOW = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
               (255, 255, 0), (255, 0, 0), (255, 0, 180)]

    def spawn(self, landmarks_px, frame_w, frame_h):
        tip = landmarks_px[LM_INDEX_TIP]
        for i in range(24):
            angle = random.uniform(0, math.tau)
            r = i * 3
            self.particles.append(Particle(
                tip[0] + math.cos(angle) * r * 0.2,
                tip[1] + math.sin(angle) * r * 0.2,
                vx=math.cos(angle) * 30, vy=math.sin(angle) * 30,
                life=self.duration * (1 - i / 30),
                size=6,
                color=self.RAINBOW[i % len(self.RAINBOW)],
            ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha) for c in p.color)
            cv2.circle(frame, (int(p.x), int(p.y)), int(p.size * (0.3 + 0.7 * alpha)),
                       color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 6. OK -> Star burst
# ---------------------------------------------------------------------------
class StarBurstEffect(ParticleEffect):
    duration = 1.3

    @staticmethod
    def _star_points(cx, cy, outer, inner, rotation=0):
        pts = []
        for i in range(10):
            r = outer if i % 2 == 0 else inner
            angle = math.radians(i * 36 + rotation)
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        return np.array(pts, dtype=np.int32)

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        for i in range(8):
            angle = (math.tau / 8) * i
            self.particles.append(Particle(
                cx, cy, vx=math.cos(angle) * 140, vy=math.sin(angle) * 140,
                life=self.duration, size=random.uniform(10, 16),
                color=(0, 220, 255),
                extra={"rot": random.uniform(0, 360)},
            ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha) for c in p.color)
            pts = self._star_points(p.x, p.y, p.size, p.size * 0.45, p.extra["rot"])
            cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 7. Love Sign -> Pink glowing particles
# ---------------------------------------------------------------------------
class PinkGlowEffect(ParticleEffect):
    duration = 1.6

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        for i in range(22):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(20, 70)
            self.particles.append(Particle(
                cx, cy, vx=math.cos(angle) * speed, vy=math.sin(angle) * speed - 20,
                ay=-10, life=random.uniform(0.8, self.duration),
                size=random.uniform(6, 14),
                color=(200, 80, 255),
            ))

    def draw(self, frame):
        glow = np.zeros_like(frame)
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha) for c in p.color)
            cv2.circle(glow, (int(p.x), int(p.y)), int(p.size), color, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=6)
        cv2.add(frame, glow, dst=frame)
        for p in self.particles:  # bright core on top
            alpha = p.life_ratio
            core = tuple(_clamp255(c * alpha) for c in (255, 220, 255))
            cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.size * 0.3)),
                       core, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 8. Fist -> Comic-style explosion
# ---------------------------------------------------------------------------
class ExplosionEffect(ParticleEffect):
    duration = 0.8

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        self.center = (cx, cy)
        for i in range(14):
            angle = (math.tau / 14) * i + random.uniform(-0.1, 0.1)
            length = random.uniform(50, 100)
            self.particles.append(Particle(
                cx, cy, vx=math.cos(angle), vy=math.sin(angle),
                life=self.duration, size=length,
                color=(0, 165, 255),
            ))

    def draw(self, frame):
        cx, cy = self.center
        for p in self.particles:
            alpha = p.life_ratio
            length = p.size * (1 - alpha * 0.3)
            x2 = int(cx + p.vx * length)
            y2 = int(cy + p.vy * length)
            x1 = int(cx + p.vx * length * 0.35)
            y1 = int(cy + p.vy * length * 0.35)
            color = tuple(_clamp255(c * alpha) for c in p.color)
            cv2.line(frame, (x1, y1), (x2, y2), color, max(2, int(6 * alpha)), cv2.LINE_AA)
        if self.particles:
            alpha = self.particles[0].life_ratio
            cv2.circle(frame, (cx, cy), int(30 * alpha) + 5,
                       (0, 255, 255), -1, cv2.LINE_AA)
            if alpha > 0.4:
                cv2.putText(frame, "POW!", (cx - 45, cy - 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.1,
                            (0, 0, 255), 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 9. Point Up -> Lightning from fingertip
# ---------------------------------------------------------------------------
class LightningEffect(ParticleEffect):
    duration = 0.5

    def spawn(self, landmarks_px, frame_w, frame_h):
        tip = landmarks_px[LM_INDEX_TIP]
        self.tip = tip
        self.bolts = []
        for _ in range(3):
            self.bolts.append(self._make_bolt(tip, frame_h))
        # dummy particle just so base class life-tracking / cleanup works
        self.particles = [Particle(tip[0], tip[1], life=self.duration)]

    def _make_bolt(self, tip, frame_h):
        points = [tip]
        x, y = tip
        target_len = random.randint(90, 160)
        steps = 7
        for i in range(1, steps + 1):
            y -= target_len / steps
            x += random.randint(-18, 18)
            points.append((int(x), int(y)))
        return points

    def draw(self, frame):
        if not self.particles:
            return
        alpha = self.particles[0].life_ratio
        color = tuple(_clamp255(c * alpha) for c in (255, 255, 120))
        for bolt in self.bolts:
            for i in range(len(bolt) - 1):
                cv2.line(frame, bolt[i], bolt[i + 1], color, 3, cv2.LINE_AA)
                cv2.line(frame, bolt[i], bolt[i + 1], (255, 255, 255),
                          1, cv2.LINE_AA)
        cv2.circle(frame, self.tip, int(10 * alpha) + 2, (255, 255, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 10. Rock Sign -> Fire effect around the hand
# ---------------------------------------------------------------------------
class FireEffect(ParticleEffect):
    duration = 1.4
    FIRE_COLORS = [(0, 60, 255), (0, 130, 255), (0, 200, 255), (60, 230, 255)]

    def spawn(self, landmarks_px, frame_w, frame_h):
        cx, cy = _palm_center(landmarks_px)
        for i in range(30):
            self.particles.append(Particle(
                cx + random.randint(-35, 35), cy + random.randint(-20, 20),
                vx=random.uniform(-10, 10), vy=random.uniform(-90, -40),
                ay=-30, life=random.uniform(0.5, self.duration),
                size=random.uniform(8, 18),
                color=random.choice(self.FIRE_COLORS),
            ))

    def draw(self, frame):
        for p in self.particles:
            alpha = p.life_ratio
            color = tuple(_clamp255(c * alpha) for c in p.color)
            size = max(1, int(p.size * alpha))
            cv2.circle(frame, (int(p.x), int(p.y)), size, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Effects manager
# ---------------------------------------------------------------------------
class EffectsManager:
    """
    Owns the mapping gesture -> effect class, keeps a list of currently
    playing effects, and enforces a per-gesture cooldown so a held gesture
    doesn't spam-retrigger every frame.
    """

    EFFECT_CLASSES = {
        "peace": BalloonEffect,
        "heart_hands": HeartsEffect,
        "thumbs_up": ConfettiEffect,
        "open_palm": SparklesEffect,
        "wave": RainbowTrailEffect,
        "ok": StarBurstEffect,
        "love_sign": PinkGlowEffect,
        "fist": ExplosionEffect,
        "point_up": LightningEffect,
        "rock_sign": FireEffect,
    }

    def __init__(self, cooldown_seconds=2.0):
        self.cooldown_seconds = cooldown_seconds
        self.active_effects = []
        self._last_trigger_time = {}

    def can_trigger(self, gesture_name):
        last = self._last_trigger_time.get(gesture_name, -math.inf)
        return (time.time() - last) >= self.cooldown_seconds

    def trigger(self, gesture_name, landmarks_px, frame_w, frame_h):
        """Attempt to start an effect. Returns True if it actually fired."""
        if gesture_name not in self.EFFECT_CLASSES:
            return False
        if not self.can_trigger(gesture_name):
            return False

        effect_cls = self.EFFECT_CLASSES[gesture_name]
        effect = effect_cls()
        effect.spawn(landmarks_px, frame_w, frame_h)
        self.active_effects.append(effect)
        self._last_trigger_time[gesture_name] = time.time()
        return True

    def update_and_draw(self, frame, dt):
        still_alive = []
        for effect in self.active_effects:
            effect.update(dt)
            effect.draw(frame)
            if effect.is_alive():
                still_alive.append(effect)
        self.active_effects = still_alive
