import math
import random
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

import pygame

# Units: AU, years, solar masses. G = 4*pi^2.
G = 4 * math.pi**2
WIDTH, HEIGHT = 1200, 800
CHART_H = int(HEIGHT * 0.20)
CX, CY = 475, CHART_H + (HEIGHT - CHART_H) // 2
DEFAULT_SCALE = 185  # pixels / AU
SCALE = DEFAULT_SCALE
ZOOM_FACTOR = 1.25
MIN_SCALE, MAX_SCALE = 35, 900
DT = 0.0006  # years per physics step
SOFTENING = 0.0008  # AU, numerical safeguard for close encounters
MAX_BODIES = 45
ENERGY_SAMPLE_EVERY = 1000  # physics steps per chart pixel
# Collision radii in AU, independent of visual pixel sizes.
STAR_COLLISION_RADIUS = 0.018
PLANET_COLLISION_SCALE = 0.00030  # planet radius (pixels) -> AU
COLLISION_RESTITUTION = 1.0  # perfectly elastic
BACKGROUND = (19, 7, 15)
WHITE = (240, 240, 240)
MUTED = (170, 170, 170)
PANEL = (33, 32, 32)  

STARS = [
    ('Red dwarf', 0.35, (255, 120, 88)),
    ('Sun-like', 1.0, (255, 219, 105)),
    ('Massive star', 2.2, (164, 199, 255)),
]
PLANETS = [
    ('Mercury', 1.66e-7, (178, 170, 166), 4),
    ('Earth', 3.0e-6, (77, 156, 250), 6),
    ('Neptune', 5.15e-5, (135, 196, 195), 8),
    ('Saturn', 2.86e-4, (222, 193, 132), 10),
    ('Jupiter', 9.55e-4, (234, 169, 119), 13),
]

@dataclass
class Body:
    name: str
    mass: float
    x: float
    y: float
    vx: float
    vy: float
    color: tuple
    radius: int
    is_star: bool = False
    trail: deque = field(default_factory=lambda: deque(maxlen=280))
    age: float = 0.0


def accelerations(bodies):
    a = [[0.0, 0.0] for _ in bodies]
    for i in range(len(bodies)):
        bi = bodies[i]
        for j in range(i + 1, len(bodies)):
            bj = bodies[j]
            dx, dy = bj.x - bi.x, bj.y - bi.y
            r2 = dx * dx + dy * dy + SOFTENING**2
            invr3 = 1.0 / (r2 * math.sqrt(r2))
            fx, fy = G * dx * invr3, G * dy * invr3
            a[i][0] += bj.mass * fx
            a[i][1] += bj.mass * fy
            a[j][0] -= bi.mass * fx
            a[j][1] -= bi.mass * fy
    return a


def integrate(bodies, dt):
    old = accelerations(bodies)
    for b, (ax, ay) in zip(bodies, old):
        b.x += b.vx * dt + 0.5 * ax * dt * dt
        b.y += b.vy * dt + 0.5 * ay * dt * dt
        b.vx += 0.5 * ax * dt
        b.vy += 0.5 * ay * dt
        b.age += dt
    new = accelerations(bodies)
    for b, (ax, ay) in zip(bodies, new):
        b.vx += 0.5 * ax * dt
        b.vy += 0.5 * ay * dt



def collision_radius(body):
    return STAR_COLLISION_RADIUS if body.is_star else body.radius * PLANET_COLLISION_SCALE


def resolve_elastic_collisions(bodies):
    """Pairwise elastic impulse and mass-weighted overlap correction.

    Collision geometry uses effective AU radii, not the drawn circle size.
    Returns a list of names for collisions with an approaching relative velocity.
    """
    events = []
    for i in range(len(bodies)):
        a = bodies[i]
        for j in range(i + 1, len(bodies)):
            b = bodies[j]
            dx, dy = b.x - a.x, b.y - a.y
            min_dist = collision_radius(a) + collision_radius(b)
            dist2 = dx * dx + dy * dy
            if dist2 >= min_dist * min_dist:
                continue
            dist = math.sqrt(dist2)
            if dist > 1e-14:
                nx, ny = dx / dist, dy / dist
            else:
                # Degenerate overlapping centers: choose relative velocity as normal.
                rvx, rvy = b.vx - a.vx, b.vy - a.vy
                norm = math.hypot(rvx, rvy)
                nx, ny = (rvx / norm, rvy / norm) if norm > 1e-14 else (1.0, 0.0)
            inv_a, inv_b = 1.0 / a.mass, 1.0 / b.mass
            inv_sum = inv_a + inv_b
            # Separate penetrated bodies to prevent repeated collision impulses.
            penetration = min_dist - dist
            correction = penetration / inv_sum
            a.x -= nx * correction * inv_a
            a.y -= ny * correction * inv_a
            b.x += nx * correction * inv_b
            b.y += ny * correction * inv_b
            relative_normal_speed = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
            if relative_normal_speed >= 0:
                continue  # already moving apart
            impulse = -(1.0 + COLLISION_RESTITUTION) * relative_normal_speed / inv_sum
            a.vx -= impulse * inv_a * nx
            a.vy -= impulse * inv_a * ny
            b.vx += impulse * inv_b * nx
            b.vy += impulse * inv_b * ny
            events.append((a.name, b.name))
    return events


def screen_xy(x, y):
    return int(CX + x * SCALE), int(CY + y * SCALE)


def world_xy(x, y):
    return (x - CX) / SCALE, (y - CY) / SCALE


def text(surface, font, s, x, y, color=WHITE):
    surface.blit(font.render(str(s), True, color), (x, y))


def energies(bodies):
    kinetic = sum(0.5 * b.mass * (b.vx*b.vx + b.vy*b.vy) for b in bodies)
    potential = 0.0
    for i, a in enumerate(bodies):
        for b in bodies[i+1:]:
            r = math.hypot(a.x-b.x, a.y-b.y)
            potential -= G * a.mass * b.mass / math.sqrt(r*r + SOFTENING**2)
    return kinetic, potential, kinetic + potential


def load_background():
    # Place your own photo next to this script as sky.jpg (or sky.png).
    folder = __import__('pathlib').Path(__file__).resolve().parent
    for filename in ('sky.jpg', 'sky.png', 'background.jpg', 'background.png'):
        path = folder / filename
        if path.is_file():
            try:
                img = pygame.image.load(str(path)).convert()
                return pygame.transform.smoothscale(img, (945, HEIGHT))
            except pygame.error:
                pass
    return None


def main():
    global SCALE  # shared by world_xy() and screen_xy()
    pygame.init()
    music_volume = 0.5
    music_enabled = False
    music_status = 'No sky.mp3'
    music_file = Path(__file__).resolve().parent / 'sky.mp3'
    if music_file.is_file():
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(music_file))
            pygame.mixer.music.set_volume(music_volume)
            pygame.mixer.music.play(-1)
            music_enabled = True
            music_status = 'Music ON'
        except (pygame.error, OSError) as exc:
            music_status = 'Music unavailable'
            print(f'Music could not start: {exc}')
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Planetary Architect v1.2 | Verlet + Zoom + Continue + Music')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('consolas', 19)
    small = pygame.font.SysFont('consolas', 15)
    big = pygame.font.SysFont('consolas', 28, bold=True)
    sky = load_background()
    backdrop = [(random.randrange(0, 945), random.randrange(HEIGHT), random.randrange(75, 175)) for _ in range(125)]
    star_idx, planet_idx = 1, 1
    bodies = []
    elapsed = 0.0
    points = 0.0
    lost = 0
    collisions = 0
    paused = False
    speed = 1
    show_trails = True
    show_bodies = True
    energy_history = []
    energy_reference = None
    chart_full = False
    physics_steps = 0
    status = 'Click on the space field to add a planet.'

    def reset():
        nonlocal bodies, elapsed, points, lost, collisions, status, energy_history, energy_reference, chart_full, paused, physics_steps
        name, mass, color = STARS[star_idx]
        bodies = [Body(name, mass, 0.0, 0.0, 0.0, 0.0, color, 21, True)]
        elapsed, points, lost, collisions = 0.0, 0.0, 0, 0
        energy_history = []
        energy_reference = None
        chart_full = False
        physics_steps = 0
        paused = False
        status = 'New star system created.'

    reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_c:
                    # Continue the same physical system with a fresh energy chart.
                    # Only the chart sampling clock is restarted; elapsed time,
                    # positions, velocities, masses, score and trails remain intact.
                    energy_history.clear()
                    energy_reference = None
                    chart_full = False
                    physics_steps = 0
                    paused = False
                    status = 'Continuing system with a fresh energy chart.'
                elif event.key == pygame.K_m:
                    if pygame.mixer.get_init() and music_file.is_file() and music_status != 'Music unavailable':
                        if music_enabled:
                            pygame.mixer.music.pause()
                            music_enabled = False
                        else:
                            pygame.mixer.music.unpause()
                            music_enabled = True
                        music_status = 'Music ON' if music_enabled else 'Music OFF'
                elif event.key == pygame.K_LEFTBRACKET:
                    music_volume = max(0.0, round(music_volume - 0.1, 2))
                    if pygame.mixer.get_init():
                        pygame.mixer.music.set_volume(music_volume)
                elif event.key == pygame.K_RIGHTBRACKET:
                    music_volume = min(1.0, round(music_volume + 0.1, 2))
                    if pygame.mixer.get_init():
                        pygame.mixer.music.set_volume(music_volume)
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_p:
                    show_bodies = not show_bodies
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    SCALE = min(MAX_SCALE, SCALE * ZOOM_FACTOR)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    SCALE = max(MIN_SCALE, SCALE / ZOOM_FACTOR)
                elif event.key in (pygame.K_0, pygame.K_KP0):
                    SCALE = DEFAULT_SCALE
                elif event.key == pygame.K_UP:
                    speed = min(16, speed * 2)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed // 2)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    star_idx = event.key - pygame.K_1
                    reset()
                elif event.key in (pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8):
                    planet_idx = event.key - pygame.K_4
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx >= 945 or my < CHART_H or chart_full:
                    continue
                if len(bodies) >= MAX_BODIES:
                    status = 'Maximum body count reached.'
                    continue
                x, y = world_xy(mx, my)
                star = bodies[0]
                dx, dy = x - star.x, y - star.y
                r = math.hypot(dx, dy)
                if r < 0.12:
                    status = 'Too close to the star!'
                    continue
                name, mass, color, radius = PLANETS[planet_idx]
                v = math.sqrt(G * (star.mass + mass) / r)
                # Tangential counterclockwise velocity relative to moving star
                vx = star.vx - v * dy / r
                vy = star.vy + v * dx / r
                bodies.append(Body(name, mass, x, y, vx, vy, color, radius))
                # Keep initial total momentum zero by giving the star the opposite impulse.
                star.vx -= mass * (vx - star.vx) / (star.mass + mass)
                star.vy -= mass * (vy - star.vy) / (star.mass + mass)
                status = f'Added {name} at {r:.2f} AU.'

        if not paused and not chart_full:
            for _ in range(speed):
                if chart_full:
                    break
                integrate(bodies, DT)
                physics_steps += 1
                elapsed += DT
                # All bodies, including the star, participate in elastic collisions.
                # No mergers or deletions on impact: momentum and kinetic energy
                # are conserved by each collision impulse (up to floating-point error).
                collision_events = resolve_elastic_collisions(bodies)
                collisions += len(collision_events)
                if collision_events:
                    a_name, b_name = collision_events[-1]
                    status = f'Elastic collision: {a_name} + {b_name}!'
                star = bodies[0]
                survivors = [star]
                for b in bodies[1:]:
                    if math.hypot(b.x - star.x, b.y - star.y) > 8.0:
                        lost += 1
                        status = f'{b.name} escaped the system!'
                    else:
                        survivors.append(b)
                bodies = survivors
                # Award survival points per simulated year, mass-dependent with diminishing returns.
                points += sum(1 + 4 * math.sqrt(b.mass / 9.55e-4) for b in bodies[1:]) * DT * 100
                if physics_steps % ENERGY_SAMPLE_EVERY == 0:
                    e = energies(bodies)
                    if energy_reference is None:
                        energy_reference = e[2]
                    energy_history.append(e)
                    if len(energy_history) >= 945 - 22:
                        chart_full = True
                        paused = True
                        status = 'Energy chart full: C continue / R reset.'
            for b in bodies:
                b.trail.append((b.x, b.y))

        screen.fill(BACKGROUND)
        if sky is not None:
            screen.blit(sky, (0, 0))
            shade_overlay = pygame.Surface((945, HEIGHT - CHART_H), pygame.SRCALPHA)
            shade_overlay.fill((20, 4, 12, 55))
            screen.blit(shade_overlay, (0, CHART_H))
        for sx, sy, shade in backdrop:
            if sy >= CHART_H and sky is None:
                pygame.draw.circle(screen, (shade, shade, shade), (sx, sy), 1)
        # Energy chart: one new x pixel every 100 physics steps.
        chart_rect = pygame.Rect(0, 0, 945, CHART_H)
        # Translucent burgundy chart on top of the Rosette Nebula photo.
        chart_overlay = pygame.Surface((945, CHART_H), pygame.SRCALPHA)
        chart_overlay.fill((32, 32, 32, 208 if sky is not None else 240))
        screen.blit(chart_overlay, (0, 0))
        pygame.draw.line(screen, (180, 180, 180), (0, CHART_H-1), (944, CHART_H-1))
        text(screen, small, 'ENERGY: kinetic / potential / total  |  1 pixel = 100 physics steps', 12, 5, MUTED)
        plot_top, plot_bottom = 29, CHART_H-13
        for fraction in (0.0, 0.5, 1.0):
            yy = int(plot_top + (plot_bottom-plot_top)*fraction)
            pygame.draw.line(screen, (180, 180, 180), (10, yy), (934, yy))
        if energy_history:
            # Shared vertical scale preserves the relative magnitudes of all energies.
            vals = [v for sample in energy_history for v in sample]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                hi, lo = hi + 1e-12, lo - 1e-12
            margin = (hi-lo)*0.08
            lo -= margin
            hi += margin
            colors = ((255, 195, 75), (95, 165, 255), (70, 230, 145))
            for component, color in enumerate(colors):
                coords = [(11+i, int(plot_bottom-(sample[component]-lo)/(hi-lo)*(plot_bottom-plot_top)))
                          for i, sample in enumerate(energy_history)]
                if len(coords) > 1:
                    pygame.draw.lines(screen, color, False, coords, 2)
            if energy_reference is not None and abs(energy_reference) > 1e-15:
                drift = (energy_history[-1][2]-energy_reference)/abs(energy_reference)*100
                text(screen, small, f'dE: {drift:+.3f}%', 790, 5, MUTED)
        # Orbital distance guides
        for au in (0.5, 1.0, 1.5, 2.0):
            guide_r = int(au * SCALE)
            pygame.draw.circle(screen, (90, 43, 64), (CX, CY), guide_r, 1)
            label_x = CX + guide_r + 3
            if label_x < 900:
                text(screen, small, f'{au:g} AU', label_x, CY + 3, MUTED)
        for b in bodies:
            if show_trails and len(b.trail) > 1:
                pts = [screen_xy(x, y) for x, y in b.trail]
                if len(pts) > 1:
                    pygame.draw.lines(screen, b.color, False, pts, 2)
            sx, sy = screen_xy(b.x, b.y)
            if show_bodies and -30 < sx < 975 and -30 < sy < HEIGHT + 30:
                if b.is_star:
                    pygame.draw.circle(screen, (75, 54, 32), (sx, sy), b.radius + 10)
                    pygame.draw.circle(screen, (135, 95, 44), (sx, sy), b.radius + 5)
                pygame.draw.circle(screen, b.color, (sx, sy), b.radius)
        pygame.draw.rect(screen, PANEL, (945, 0, 255, HEIGHT))
        pygame.draw.line(screen, (180, 180, 180), (945, 0), (945, HEIGHT), 2)
        text(screen, big, 'PLANETARY', 961, 18)
        text(screen, big, 'ARCHITECT', 961, 49)
        text(screen, small, 'v1.2 | Verlet + music', 961, 90, MUTED)
        text(screen, font, 'STAR  (1-3)', 961, 131)
        for i, (name, mass, _) in enumerate(STARS):
            text(screen, small, f'{">" if i == star_idx else " "} {i+1} {name}', 961, 162 + i * 25,
                 WHITE if i == star_idx else MUTED)
        text(screen, font, 'PLANET  (4-8)', 961, 259)
        for i, (name, mass, _, _) in enumerate(PLANETS):
            text(screen, small, f'{">" if i == planet_idx else " "} {i+4} {name}', 961, 291 + i * 25,
                 WHITE if i == planet_idx else MUTED)
        text(screen, font, 'SIMULATION', 961, 447)
        text(screen, small, f'Time: {elapsed:.2f} years', 961, 479)
        text(screen, small, f'Speed: x{speed} | Zoom: {SCALE / DEFAULT_SCALE:.2f}x', 961, 505)
        text(screen, small, f'Planets: {len(bodies)-1}', 961, 531)
        text(screen, small, f'Escaped: {lost}', 961, 557)
        text(screen, small, f'Collisions: {collisions}', 961, 583)
        text(screen, font, f'SCORE: {int(points)}', 961, 625)
        text(screen, small, 'SPACE pause | R reset | C next', 961, 673, MUTED)
        text(screen, small, 'UP/DOWN speed | T trails', 961, 699, MUTED)
        text(screen, small, 'P bodies | +/- zoom | 0 fit', 961, 725, MUTED)
        text(screen, small, f'M toggle | [/] vol: {int(music_volume * 100)}%', 961, 751, MUTED)
        text(screen, small, music_status, 961, 773, MUTED)
        text(screen, small, ('FINISHED - C continue / R reset' if chart_full else 'PAUSED' if paused else status[:60]), 18, HEIGHT - 30)
        pygame.display.flip()
        clock.tick(60)
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
    pygame.quit()


if __name__ == '__main__':
    main()
