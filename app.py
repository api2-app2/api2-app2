from flask import Flask, send_from_directory, request, jsonify
import math

app = Flask(__name__, static_folder=".", static_url_path="")

def _mod360(x: float) -> float:
    return (x % 360.0 + 360.0) % 360.0

def _ang_diff_180(a: float, b: float) -> float:
    """
    Signed shortest angular difference a-b in [-180, 180).
    """
    d = _mod360(a - b)
    if d >= 180.0:
        d -= 360.0
    return d

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _smooth_sign(x: float, width_deg: float) -> float:
    """
    Smooth sign in [-1, +1], continuous through 0.
    """
    w = float(width_deg)
    if w <= 1e-9:
        return 0.0 if abs(x) < 1e-12 else (1.0 if x > 0 else -1.0)
    return math.tanh(x / w)

def _base_blinds_closed_pct(sun_az: float, wall_normal_az: float) -> float:
    """
    Base tracking (monotonic mapping) from your original logic:
      wallDirAz      = mod360(wallNormalAz + 90)
      targetBlindDir = mod360(sunAz - 90)
      delta          = ((targetBlindDir - wallDirAz) % 180)   # 0..180
      pct            = (delta / 180) * 100
    """
    wall_dir = _mod360(wall_normal_az + 90.0)
    target_bdir = _mod360(sun_az - 90.0)
    delta = ((target_bdir - wall_dir) % 180.0)  # 0..180
    return _clamp((delta / 180.0) * 100.0, 0.0, 100.0)

def blinds_closed_pct(
    sun_az: float,
    wall_normal_az: float,
    offset_mag_pct: float = 0.0,
    smooth_width_deg: float = 10.0,
    edge_buffer_pct: float = 10.0,
):
    """
    Returns:
      (pct_final, pct_base, off_applied, pct_pre_clamp, min_allowed, max_allowed)

    Pipeline:
      1) pct_base from tracking geometry
      2) off_applied = offset_mag * tanh(diff/width)  (diff = sun-wall, signed shortest)
      3) pct_pre_clamp = clamp(pct_base + off_applied, 0..100)
      4) symmetric edge clamp:
         pct_final = clamp(pct_pre_clamp, edge_buffer_pct, 100-edge_buffer_pct)

    This enforces: NOT > max_allowed and NOT < min_allowed.
    """
    sun = _mod360(float(sun_az))
    wall = _mod360(float(wall_normal_az))

    offset_mag = max(0.0, float(offset_mag_pct))
    width = max(0.0, float(smooth_width_deg))

    # keep edge buffer sane (avoid inverted bounds)
    edge = _clamp(float(edge_buffer_pct), 0.0, 49.9)
    min_allowed = edge
    max_allowed = 100.0 - edge

    base = _base_blinds_closed_pct(sun, wall)
    diff = _ang_diff_180(sun, wall)  # signed shortest sun-wall

    off_applied = offset_mag * (0.0 if abs(diff) < 1e-9 else (1.0 if diff > 0 else -1.0))
    pre = _clamp(base + off_applied, 0.0, 100.0)

    final = _clamp(pre, min_allowed, max_allowed)
    return final, base, off_applied, pre, min_allowed, max_allowed

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/api/pct")
def api_pct():
    sun = float(request.args.get("sun", "0"))
    wall = float(request.args.get("wall", "0"))
    offset = float(request.args.get("offset", "0"))
    width = float(request.args.get("width", "10"))
    edge = float(request.args.get("edge", "10"))

    pct_final, pct_base, off_applied, pct_pre, mn, mx = blinds_closed_pct(
        sun, wall, offset, width, edge
    )

    return jsonify({
        "sun": sun,
        "wall": wall,
        "offset": offset,
        "width": width,
        "edge": edge,
        "min_allowed": mn,
        "max_allowed": mx,
        "pct": pct_final,
        "pct_pre": pct_pre,
        "pct_base": pct_base,
        "offset_applied": off_applied,
    })

@app.get("/api/table")
def api_table():
    wall = float(request.args.get("wall", "0"))
    offset = float(request.args.get("offset", "0"))
    width = float(request.args.get("width", "10"))
    edge = float(request.args.get("edge", "10"))

    final_table = []
    base_table = []
    off_table = []
    pre_table = []

    # bounds are constant for this request
    edge_clamped = _clamp(edge, 0.0, 49.9)
    min_allowed = edge_clamped
    max_allowed = 100.0 - edge_clamped

    for s in range(360):
        pct_final, pct_base, off_applied, pct_pre, _, _ = blinds_closed_pct(
            float(s), wall, offset, width, edge
        )
        final_table.append(pct_final)
        base_table.append(pct_base)
        off_table.append(off_applied)
        pre_table.append(pct_pre)

    return jsonify({
        "wall": wall,
        "offset": offset,
        "width": width,
        "edge": edge,
        "min_allowed": min_allowed,
        "max_allowed": max_allowed,
        "table": final_table,      # FINAL % (after symmetric edge clamp)
        "pre_table": pre_table,    # before symmetric edge clamp
        "base_table": base_table,
        "offset_applied": off_table
    })

if __name__ == "__main__":
    # Hugging Face Spaces expects port 7860
    app.run(host="0.0.0.0", port=7860)
