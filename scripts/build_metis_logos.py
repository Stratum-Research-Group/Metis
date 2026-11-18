#!/usr/bin/env python3
"""
Generate 7 distinct, precision Metis logos matching the dimensions, typography,
layout, and studio line-art aesthetic of dioptra_logo.jpg (1376x768).

Concepts:
  01 - Kybernetes Rudder & Celestial Helm (Steering & Orchestration)
  02 - Poincaré Disk & Hyperbolic Foliation (Non-Euclidean Memory)
  03 - Deliberative Thought-Graph & Sheaf Lattice (Cognitive Graph & MCTS)
  04 - Lorentzian Light Cone & Causal Horizon (Causal As-Of Temporal Bounds)
  05 - Geometric Owl of Metis (Wisdom, Counsel & Technical Craft)
  06 - Kleroterion Stochastic Dial & Decision Horizon (Stochastic Decision & Conformal Gating)
  07 - Phylax Hexagonal Bastion & Cryptographic Shield (Guarded Capability Gateway)
"""

import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions
WIDTH = 1376
HEIGHT = 768
SCALE = 2  # 2x supersampling for ultra-crisp anti-aliasing
W_S = WIDTH * SCALE
H_S = HEIGHT * SCALE

# Typography & Layout anchors (in 1x coordinates)
DIVIDER_X = 631
DIVIDER_TOP = 185
DIVIDER_BOT = 582
DIVIDER_WIDTH = 11

TEXT_X = 688
TITLE_Y = 184  # Font size 178 puts cap-top at y=216
SUB_X = 690
SUB_Y1 = 354  # "Stratum"
SUB_Y2 = 429  # "Research"
SUB_Y3 = 504  # "Group"

EMBLEM_CX = 390
EMBLEM_CY = 384

FONT_PATH = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/noto/NotoSans-Bold.ttf"


def create_studio_background(w, h, seed=42):
    """Generate subtle studio paper off-white background with micro-grain noise."""
    rng = np.random.default_rng(seed)
    y_grad = np.linspace(231.8, 230.5, h)[:, None]
    x_grad = np.linspace(231.2, 230.8, w)[None, :]
    base_luma = (y_grad + x_grad) / 2.0

    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = base_luma
    bg[:, :, 1] = base_luma
    bg[:, :, 2] = base_luma - 1.2  # subtle warm paper undertone

    noise = rng.normal(0.0, 1.8, size=(h, w, 3))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(bg)


def render_base_template(draw, scale=SCALE):
    """Render the vertical divider bar and standard text."""
    # Divider line
    div_x = DIVIDER_X * scale
    div_w = DIVIDER_WIDTH * scale
    y0 = DIVIDER_TOP * scale
    y1 = DIVIDER_BOT * scale
    draw.rounded_rectangle([div_x - div_w // 2, y0, div_x + div_w // 2, y1], radius=scale, fill=(0, 0, 0, 255))

    # Fonts
    font_title = ImageFont.truetype(FONT_PATH, int(178 * scale))
    font_sub = ImageFont.truetype(FONT_PATH, int(76 * scale))

    # Text
    draw.text((int(TEXT_X * scale), int(TITLE_Y * scale)), "Metis", font=font_title, fill=(0, 0, 0, 255))
    draw.text((int(SUB_X * scale), int(SUB_Y1 * scale)), "Stratum", font=font_sub, fill=(0, 0, 0, 255))
    draw.text((int(SUB_X * scale), int(SUB_Y2 * scale)), "Research", font=font_sub, fill=(0, 0, 0, 255))
    draw.text((int(SUB_X * scale), int(SUB_Y3 * scale)), "Group", font=font_sub, fill=(0, 0, 0, 255))


# ---------------------------------------------------------------------------
# EMBLEM GENERATORS (Render at 2x scale directly onto transparent overlay)
# ---------------------------------------------------------------------------

def draw_concept_01_rudder_helm(overlay, cx, cy, s):
    """Concept 01: Kybernetes Rudder & Celestial Helm."""
    draw = ImageDraw.Draw(overlay)
    # Outer navigation wheel rim
    r_outer = 152 * s
    r_inner = 110 * s
    r_hub = 32 * s
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(0, 0, 0, 255), width=int(10 * s))
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], outline=(0, 0, 0, 255), width=int(6 * s))
    draw.ellipse([cx - r_hub, cy - r_hub, cx + r_hub, cy + r_hub], outline=(0, 0, 0, 255), width=int(8 * s))
    draw.ellipse([cx - 10 * s, cy - 10 * s, cx + 10 * s, cy + 10 * s], fill=(0, 0, 0, 255))

    # 8 radial spokes, 4 cardinal spokes extend past outer rim
    for i in range(8):
        angle = i * (math.pi / 4.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        is_cardinal = (i % 2 == 0)
        r_reach = (r_outer + 28 * s) if is_cardinal else r_outer
        w_spoke = int(8 * s) if is_cardinal else int(5 * s)
        draw.line([cx + r_hub * cos_a, cy + r_hub * sin_a, cx + r_reach * cos_a, cy + r_reach * sin_a], fill=(0, 0, 0, 255), width=w_spoke)
        if is_cardinal:
            # Spoke handle node
            hx = cx + r_reach * cos_a
            hy = cy + r_reach * sin_a
            draw.ellipse([hx - 7 * s, hy - 7 * s, hx + 7 * s, hy + 7 * s], fill=(0, 0, 0, 255))

    # Sweeping directional rudder keel arc slicing through lower-right
    rudder_box = [cx - 180 * s, cy - 60 * s, cx + 180 * s, cy + 195 * s]
    draw.arc(rudder_box, start=30, end=115, fill=(0, 0, 0, 255), width=int(12 * s))

    # Calibration ticks along the inner arc
    for deg in range(15, 75, 12):
        rad = math.radians(deg)
        x_in = cx + (r_inner - 14 * s) * math.cos(rad)
        y_in = cy + (r_inner - 14 * s) * math.sin(rad)
        x_out = cx + (r_inner + 14 * s) * math.cos(rad)
        y_out = cy + (r_inner + 14 * s) * math.sin(rad)
        draw.line([x_in, y_in, x_out, y_out], fill=(0, 0, 0, 255), width=int(4 * s))


def draw_concept_02_poincare_hyperbolic(overlay, cx, cy, s):
    """Concept 02: Poincaré Disk & Hyperbolic Foliation (Strictly Clipped)."""
    r = 152 * s

    # 1. Create separate layer for interior geodesics
    geo_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    geo_draw = ImageDraw.Draw(geo_layer)

    # Straight orthogonal diameters (hyperbolic geodesics through the center)
    geo_draw.line([cx - r, cy, cx + r, cy], fill=(0, 0, 0, 255), width=int(8 * s))
    geo_draw.line([cx, cy - r, cx, cy + r], fill=(0, 0, 0, 255), width=int(8 * s))

    # Orthogonal circular arc geodesics intersecting the boundary circle at 90 degrees
    # Circle centered at distance d from origin has radius R = sqrt(d^2 - r^2)
    for d_factor in [1.25, 1.60, 2.10]:
        d = d_factor * r
        R = math.sqrt(d**2 - r**2)
        for sign in [-1, 1]:
            # Horizontal offset circles
            ox = cx + sign * d
            geo_draw.ellipse([ox - R, cy - R, ox + R, cy + R], outline=(0, 0, 0, 255), width=int(6 * s))
            # Vertical offset circles
            oy = cy + sign * d
            geo_draw.ellipse([cx - R, oy - R, cx + R, oy + R], outline=(0, 0, 0, 255), width=int(6 * s))

    # Diagonal horocycles
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            d_diag = 1.45 * r
            R_diag = math.sqrt(d_diag**2 - r**2)
            diag_x = cx + sign_x * (d_diag / math.sqrt(2))
            diag_y = cy + sign_y * (d_diag / math.sqrt(2))
            geo_draw.ellipse([diag_x - R_diag, diag_y - R_diag, diag_x + R_diag, diag_y + R_diag], outline=(0, 0, 0, 255), width=int(5 * s))

    # 2. Circular mask to strictly clip geodesics to interior of Poincaré disk
    mask = Image.new("L", overlay.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([cx - r + 3*s, cy - r + 3*s, cx + r - 3*s, cy + r - 3*s], fill=255)

    # Composite clipped geodesics onto overlay
    overlay.paste(geo_layer, (0, 0), mask)

    # 3. Draw boundary rim and center metric node on top
    draw = ImageDraw.Draw(overlay)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0, 255), width=int(11 * s))
    draw.ellipse([cx - 40 * s, cy - 40 * s, cx + 40 * s, cy + 40 * s], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(6 * s))
    draw.ellipse([cx - 14 * s, cy - 14 * s, cx + 14 * s, cy + 14 * s], fill=(0, 0, 0, 255))


def draw_concept_03_deliberative_graph(overlay, cx, cy, s):
    """Concept 03: Deliberative Thought-Graph & Sheaf Lattice."""
    draw = ImageDraw.Draw(overlay)
    r_hex = 145 * s
    # Generate 6 hexagonal outer vertices + 1 center node
    nodes = [(cx, cy)]
    for i in range(6):
        a = i * (math.pi / 3.0) - (math.pi / 6.0)
        nodes.append((cx + r_hex * math.cos(a), cy + r_hex * math.sin(a)))

    # Outer perimeter edges
    for i in range(1, 7):
        next_i = (i % 6) + 1
        draw.line([nodes[i], nodes[next_i]], fill=(0, 0, 0, 255), width=int(6 * s))

    # Radial spokes to center
    for i in range(1, 7):
        draw.line([nodes[0], nodes[i]], fill=(0, 0, 0, 255), width=int(5 * s))

    # Primary execution path (thick highlighted decision trace: 4 -> 0 -> 2 -> 1)
    primary_path = [nodes[4], nodes[0], nodes[2], nodes[1]]
    for i in range(len(primary_path) - 1):
        draw.line([primary_path[i], primary_path[i+1]], fill=(0, 0, 0, 255), width=int(14 * s))

    # Epistemic sheaf tangent arcs across cross-branches
    draw.arc([nodes[0][0] - 80 * s, nodes[0][1] - 80 * s, nodes[0][0] + 80 * s, nodes[0][1] + 80 * s], start=30, end=210, fill=(0, 0, 0, 255), width=int(5 * s))

    # Draw nodes
    for i, (nx, ny) in enumerate(nodes):
        if i == 0:
            # Hub node
            draw.ellipse([nx - 24 * s, ny - 24 * s, nx + 24 * s, ny + 24 * s], fill=(0, 0, 0, 255))
            draw.ellipse([nx - 10 * s, ny - 10 * s, nx + 10 * s, ny + 10 * s], fill=(232, 232, 230, 255))
        elif i in [1, 2, 4]:
            # Nodes on active thought trajectory
            draw.ellipse([nx - 19 * s, ny - 19 * s, nx + 19 * s, ny + 19 * s], fill=(0, 0, 0, 255))
        else:
            # Standard boundary nodes
            draw.ellipse([nx - 16 * s, ny - 16 * s, nx + 16 * s, ny + 16 * s], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(5 * s))


def draw_concept_04_lorentz_cone(overlay, cx, cy, s):
    """Concept 04: Lorentzian Light Cone & Causal Horizon."""
    draw = ImageDraw.Draw(overlay)
    h_cone = 150 * s
    w_top = 135 * s
    rx_top = w_top
    ry_top = 34 * s

    # Top future spatial slice ellipse
    draw.ellipse([cx - rx_top, cy - h_cone - ry_top, cx + rx_top, cy - h_cone + ry_top], outline=(0, 0, 0, 255), width=int(9 * s))
    # Bottom past spatial slice ellipse
    draw.ellipse([cx - rx_top, cy + h_cone - ry_top, cx + rx_top, cy + h_cone + ry_top], outline=(0, 0, 0, 255), width=int(9 * s))

    # Diagonal null-geodesic asymptotes meeting at the apex (cx, cy)
    draw.line([cx - w_top, cy - h_cone, cx + w_top, cy + h_cone], fill=(0, 0, 0, 255), width=int(10 * s))
    draw.line([cx + w_top, cy - h_cone, cx - w_top, cy + h_cone], fill=(0, 0, 0, 255), width=int(10 * s))

    # Intermediate wave propagation ellipses (nested intermediate time slices)
    mid_scale = 0.52
    h_mid = h_cone * mid_scale
    rx_mid = w_top * mid_scale
    ry_mid = ry_top * mid_scale
    draw.ellipse([cx - rx_mid, cy - h_mid - ry_mid, cx + rx_mid, cy - h_mid + ry_mid], outline=(0, 0, 0, 255), width=int(5 * s))
    draw.ellipse([cx - rx_mid, cy + h_mid - ry_mid, cx + rx_mid, cy + h_mid + ry_mid], outline=(0, 0, 0, 255), width=int(5 * s))

    # Horizontal simultaneity plane through the singularity
    draw.line([cx - 160 * s, cy, cx + 160 * s, cy], fill=(0, 0, 0, 255), width=int(6 * s))

    # Central observer singularity (apex ring and core)
    draw.ellipse([cx - 22 * s, cy - 22 * s, cx + 22 * s, cy + 22 * s], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(6 * s))
    draw.ellipse([cx - 8 * s, cy - 8 * s, cx + 8 * s, cy + 8 * s], fill=(0, 0, 0, 255))


def draw_concept_05_athena_owl(overlay, cx, cy, s):
    """Concept 05: Sleek Geometric Owl of Metis (Greek Wisdom & Architectural Craft)."""
    draw = ImageDraw.Draw(overlay)

    # Outer circular seal border
    r_seal = 152 * s
    draw.ellipse([cx - r_seal, cy - r_seal, cx + r_seal, cy + r_seal], outline=(0, 0, 0, 255), width=int(10 * s))

    # Triangular crest ears at top
    ear_l = [(cx - 95 * s, cy - 85 * s), (cx - 40 * s, cy - 135 * s), (cx - 15 * s, cy - 80 * s)]
    ear_r = [(cx + 15 * s, cy - 80 * s), (cx + 40 * s, cy - 135 * s), (cx + 95 * s, cy - 85 * s)]
    draw.polygon(ear_l, outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(7 * s))
    draw.polygon(ear_r, outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(7 * s))

    # Dual concentric optical lens eyes
    eye_dx = 48 * s
    eye_cy = cy - 20 * s
    r_out = 46 * s
    r_mid = 28 * s
    r_in = 12 * s

    for sign in [-1, 1]:
        ex = cx + sign * eye_dx
        draw.ellipse([ex - r_out, eye_cy - r_out, ex + r_out, eye_cy + r_out], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(8 * s))
        draw.ellipse([ex - r_mid, eye_cy - r_mid, ex + r_mid, eye_cy + r_mid], outline=(0, 0, 0, 255), width=int(4 * s))
        draw.ellipse([ex - r_in, eye_cy - r_in, ex + r_in, eye_cy + r_in], fill=(0, 0, 0, 255))

    # Sharp geometric beak
    beak_pts = [
        (cx - 16 * s, eye_cy + 8 * s),
        (cx + 16 * s, eye_cy + 8 * s),
        (cx, eye_cy + 55 * s)
    ]
    draw.polygon(beak_pts, fill=(0, 0, 0, 255))

    # Sleek angular wing mantle framing the chest
    draw.line([cx - eye_dx - r_out, eye_cy, cx - 45 * s, cy + 95 * s], fill=(0, 0, 0, 255), width=int(8 * s))
    draw.line([cx + eye_dx + r_out, eye_cy, cx + 45 * s, cy + 95 * s], fill=(0, 0, 0, 255), width=int(8 * s))
    draw.line([cx - 45 * s, cy + 95 * s, cx, cy + 120 * s], fill=(0, 0, 0, 255), width=int(8 * s))
    draw.line([cx + 45 * s, cy + 95 * s, cx, cy + 120 * s], fill=(0, 0, 0, 255), width=int(8 * s))

    # Center breastplate chevron line
    draw.line([cx, eye_cy + 55 * s, cx, cy + 120 * s], fill=(0, 0, 0, 255), width=int(6 * s))


def draw_concept_06_entropy_calibrator(overlay, cx, cy, s):
    """Concept 06: Kleroterion Stochastic Dial & Decision Horizon."""
    draw = ImageDraw.Draw(overlay)
    r = 150 * s

    # Semi-circular graduated protractor arc (open top)
    draw.arc([cx - r, cy - r, cx + r, cy + r], start=150, end=390, fill=(0, 0, 0, 255), width=int(10 * s))
    draw.arc([cx - 0.72 * r, cy - 0.72 * r, cx + 0.72 * r, cy + 0.72 * r], start=150, end=390, fill=(0, 0, 0, 255), width=int(5 * s))

    # Graduated radial vernier entropy ticks
    for deg in range(160, 381, 10):
        rad = math.radians(deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        tick_len = (24 * s) if (deg % 30 == 0) else (14 * s)
        w_tick = int(6 * s) if (deg % 30 == 0) else int(3.5 * s)
        draw.line([cx + (r - tick_len) * cos_r, cy + (r - tick_len) * sin_r, cx + r * cos_r, cy + r * sin_r], fill=(0, 0, 0, 255), width=w_tick)

    # Balance horizon baseline
    draw.line([cx - 165 * s, cy + 50 * s, cx + 165 * s, cy + 50 * s], fill=(0, 0, 0, 255), width=int(8 * s))

    # Central pivot needle poised at active decision angle (-65 degrees)
    needle_angle = math.radians(-70)
    nx = cx + 135 * s * math.cos(needle_angle)
    ny = cy + 135 * s * math.sin(needle_angle)
    draw.line([cx, cy + 50 * s, nx, ny], fill=(0, 0, 0, 255), width=int(9 * s))
    # Needle tip arrow
    draw.ellipse([nx - 9 * s, ny - 9 * s, nx + 9 * s, ny + 9 * s], fill=(0, 0, 0, 255))

    # Central pivot mechanism
    draw.ellipse([cx - 28 * s, cy + 50 * s - 28 * s, cx + 28 * s, cy + 50 * s + 28 * s], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(7 * s))
    draw.ellipse([cx - 10 * s, cy + 50 * s - 10 * s, cx + 10 * s, cy + 50 * s + 10 * s], fill=(0, 0, 0, 255))

    # Symmetrical dual balance pendulum weights hanging below
    for sign in [-1, 1]:
        px = cx + sign * 115 * s
        draw.line([px, cy + 50 * s, px, cy + 115 * s], fill=(0, 0, 0, 255), width=int(5 * s))
        draw.ellipse([px - 16 * s, cy + 115 * s - 16 * s, px + 16 * s, cy + 115 * s + 16 * s], fill=(0, 0, 0, 255))


def draw_concept_07_phylax_shield(overlay, cx, cy, s):
    """Concept 07: Phylax Hexagonal Bastion & Cryptographic Shield."""
    draw = ImageDraw.Draw(overlay)
    r_outer = 152 * s
    r_inner = 114 * s

    def get_hex_points(radius, angle_offset=0):
        pts = []
        for i in range(6):
            a = i * (math.pi / 3.0) + angle_offset
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        return pts

    pts_outer = get_hex_points(r_outer)
    pts_inner = get_hex_points(r_inner)

    # Outer and inner regular hexagons
    for i in range(6):
        next_i = (i + 1) % 6
        draw.line([pts_outer[i], pts_outer[next_i]], fill=(0, 0, 0, 255), width=int(10 * s))
        draw.line([pts_inner[i], pts_inner[next_i]], fill=(0, 0, 0, 255), width=int(6 * s))

    # Tri-fold Y-axis symmetry dividing interior into 3 shielded sectors
    for i in [0, 2, 4]:
        draw.line([cx, cy, pts_inner[i][0], pts_inner[i][1]], fill=(0, 0, 0, 255), width=int(9 * s))

    # Perimeter bastion anchor nodes at outer vertices
    for i in range(6):
        px, py = pts_outer[i]
        draw.ellipse([px - 12 * s, py - 12 * s, px + 12 * s, py + 12 * s], fill=(0, 0, 0, 255))

    # Central cryptographic keyhole aperture
    r_core = 36 * s
    draw.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(8 * s))
    draw.ellipse([cx - 14 * s, cy - 14 * s, cx + 14 * s, cy + 14 * s], fill=(0, 0, 0, 255))


# Concept Registry
CONCEPTS = [
    {
        "id": "metis_logo_01_rudder_helm",
        "title": "Kybernetes Rudder & Celestial Helm",
        "desc": "Nautical steersman helm, governor rudder, and cardinal compass spokes (Kybernetes control plane & steering)",
        "draw_fn": draw_concept_01_rudder_helm,
    },
    {
        "id": "metis_logo_02_poincare_hyperbolic",
        "title": "Poincaré Disk & Geodesic Foliation",
        "desc": "Hyperbolic disk with orthogonal metric geodesics and horocycles (LTHCM non-Euclidean memory)",
        "draw_fn": draw_concept_02_poincare_hyperbolic,
    },
    {
        "id": "metis_logo_03_deliberative_graph",
        "title": "Deliberative Thought-Graph & Sheaf Lattice",
        "desc": "Hexagonal hypothesis graph with primary execution trace and sheaf contradiction branches (Dianoia MCTS)",
        "draw_fn": draw_concept_03_deliberative_graph,
    },
    {
        "id": "metis_logo_04_lorentz_cone",
        "title": "Lorentzian Light Cone & Causal Horizon",
        "desc": "Hourglass double light-cone with tilted spatial-slice ellipses and simultaneity hyperplane (Causal bounds)",
        "draw_fn": draw_concept_04_lorentz_cone,
    },
    {
        "id": "metis_logo_05_athena_owl",
        "title": "Geometric Owl of Metis",
        "desc": "Architectural geometric owl seal with dual optical lens eyes, crest horns, and folded mantle (Wisdom & craft)",
        "draw_fn": draw_concept_05_athena_owl,
    },
    {
        "id": "metis_logo_06_entropy_calibrator",
        "title": "Kleroterion Stochastic Dial & Decision Horizon",
        "desc": "Graduated vernier goniometer dial, threshold indicator needle, and balance pendulums (Stochastic policy & conformal gates)",
        "draw_fn": draw_concept_06_entropy_calibrator,
    },
    {
        "id": "metis_logo_07_phylax_shield",
        "title": "Phylax Hexagonal Bastion & Cryptographic Shield",
        "desc": "Double-perimeter bastion hexagon with triquetra partition and central keyhole aperture (Phylax capability gateway)",
        "draw_fn": draw_concept_07_phylax_shield,
    },
]


def main():
    assets_dir = Path("/home/akarta/Projects/Metis/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    branding_dir = Path("/home/akarta/Projects/z.branding")

    print(f"[*] Building 7 precision Metis logos at {WIDTH}x{HEIGHT} (2x supersampled to {W_S}x{H_S})...")

    # Pre-generate base studio texture at 1x
    base_bg = create_studio_background(WIDTH, HEIGHT, seed=108)

    for i, c in enumerate(CONCEPTS):
        print(f"[{i+1}/7] Rendering {c['id']}: {c['title']}...")

        # Create 2x transparent overlay
        overlay = Image.new("RGBA", (W_S, H_S), (0, 0, 0, 0))

        # 1. Render Left Emblem (passes overlay so clipping can be used)
        c["draw_fn"](overlay, EMBLEM_CX * SCALE, EMBLEM_CY * SCALE, SCALE)

        # 2. Render layout template (Divider + Typography) on top
        draw_overlay = ImageDraw.Draw(overlay)
        render_base_template(draw_overlay, scale=SCALE)

        # 3. Downsample overlay with Lanczos anti-aliasing
        overlay_1x = overlay.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

        # 4. Composite onto studio paper background
        composite = base_bg.copy()
        composite.paste(overlay_1x, (0, 0), overlay_1x)

        # 5. Save outputs: high-quality JPEG (matching dioptra_logo.jpg) and lossless PNG
        jpg_path = assets_dir / f"{c['id']}.jpg"
        png_path = assets_dir / f"{c['id']}.png"
        composite.save(jpg_path, quality=98, subsampling=0)
        composite.save(png_path)

        # Also copy to z.branding for centralized asset access
        if branding_dir.is_dir():
            import shutil
            shutil.copy(jpg_path, branding_dir / jpg_path.name)
            shutil.copy(png_path, branding_dir / png_path.name)

        print(f"      -> Saved {jpg_path} (and {png_path})")

    print("[+] All 7 logos rendered successfully!")


if __name__ == "__main__":
    main()
