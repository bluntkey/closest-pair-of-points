"""
Closest Pair of Points: Brute-Force vs Divide-and-Conquer
Usage:  python closest_pair.py            (reads N from stdin)
        python closest_pair.py 500        (N from argv)
Outputs: closest_pair_points.png, closest_pair_runtime.png, closest_pair_animation.gif
"""
import sys
import math
import random
import time

import matplotlib
matplotlib.use("Agg")  # works without a display; remove for interactive windows
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ---------------------------------------------------------------- Brute force
def brute_force(pts):
    """O(n^2). Returns (d, p, q) or None if fewer than 2 points."""
    n = len(pts)
    if n < 2:
        return None
    best = (dist(pts[0], pts[1]), pts[0], pts[1])
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(pts[i], pts[j])
            if d < best[0]:
                best = (d, pts[i], pts[j])
    return best


# -------------------------------------------------------- Divide and conquer
def divide_and_conquer(pts, info=None, trace=None):
    """O(n log n). `info` (dict) receives the top-level split line,
    the strip half-width and the strip points, for plotting."""
    n = len(pts)
    if n < 2:
        return None
    px = sorted(pts)                          # sorted by x (then y)
    py = sorted(pts, key=lambda p: (p[1], p[0]))  # sorted by y
    return _rec(px, py, info, trace, top=True)


def _rec(px, py, info, trace=None, top=False):
    n = len(px)
    if n <= 3:                                # base case
        res = brute_force(px)
        if trace is not None:
            trace.append(dict(kind="base", pts=px, best=res))
        return res

    mid = n // 2
    mid_pt = px[mid]
    if trace is not None:
        trace.append(dict(kind="split", pts=px, line_x=mid_pt[0]))
    left_x, right_x = px[:mid], px[mid:]
    left_set = set(map(id, left_x))
    # split py into left/right halves while keeping y-order.
    # (identity-based so duplicate points are handled correctly)
    left_y, right_y = [], []
    for p in py:
        (left_y if id(p) in left_set else right_y).append(p)

    dl = _rec(left_x, left_y, None, trace)
    dr = _rec(right_x, right_y, None, trace)
    best = dl if dl[0] <= dr[0] else dr
    d = best[0]

    # strip: points within d of the dividing line x = mid_pt[0]
    line_x = mid_pt[0]
    strip = [p for p in py if abs(p[0] - line_x) < d]

    # each point only needs comparing with the next few (<= 7) in y-order
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and strip[j][1] - strip[i][1] < best[0]:
            dd = dist(strip[i], strip[j])
            if dd < best[0]:
                best = (dd, strip[i], strip[j])
            j += 1

    if trace is not None:
        trace.append(dict(kind="merge", pts=px, line_x=line_x, d=d,
                          strip=strip, best=best, left=dl, right=dr))
    if top and info is not None:
        info["line_x"] = line_x
        info["d_before_strip"] = d
        info["strip"] = strip
    return best


# ------------------------------------------------------------------- helpers
def timed(fn, *args):
    t = time.perf_counter()
    r = fn(*args)
    return r, (time.perf_counter() - t) * 1000.0


def fmt(res):
    if res is None:
        return "N/A (need at least 2 points)"
    d, p, q = res
    return f"{d:.6f} ({p[0]:.6f}, {p[1]:.6f}) ({q[0]:.6f}, {q[1]:.6f})"


def gen_points(n, lo=0, hi=100, float_pts=True):
    if float_pts:
        return [(random.uniform(lo, hi), random.uniform(lo, hi)) for _ in range(n)]
    return [(random.randint(lo, hi), random.randint(lo, hi)) for _ in range(n)]


# --------------------------------------------------------------- verification
def run_tests():
    ok = True
    cases = {
        "N=0": [], "N=1": [(1, 1)], "N=2": [(0, 0), (3, 4)],
        "duplicates": [(1, 1), (5, 5), (1, 1), (9, 2)],
        "all identical": [(2, 2)] * 6,
        "same x": [(3, y) for y in (0, 10, 4, 7, 1.5)],
        "same y": [(x, 3) for x in (0, 10, 4, 7, 1.5)],
        "collinear diag": [(i, i) for i in (0, 5, 2, 9, 3.5, 8)],
        "negatives": [(-5, -5), (-4, -5.5), (10, 10), (-20, 3)],
    }
    for name, pts in cases.items():
        b, d = brute_force(pts), divide_and_conquer(pts)
        same = (b is None and d is None) or (
            b is not None and d is not None and abs(b[0] - d[0]) < 1e-9)
        ok &= same
        print(f"  [{'OK' if same else 'FAIL'}] {name}")

    random.seed(1)
    for t in range(2000):                      # randomised cross-check
        n = random.randint(2, 60)
        pts = gen_points(n, 0, random.choice([3, 10, 1000]),
                         float_pts=random.random() < 0.5)  # small ranges => many dups/ties
        b, d = brute_force(pts), divide_and_conquer(pts)
        if abs(b[0] - d[0]) > 1e-9:
            ok = False
            print("  [FAIL] random mismatch", pts)
            break
    print(f"  [{'OK' if ok else 'FAIL'}] 2000 random cross-checks")
    return ok


# ------------------------------------------------------------------- plotting
def plot_points(pts, bf, dc, info, fname="closest_pair_points.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=14, c="gray", alpha=0.7, label="points")

    if info:
        lx, d = info["line_x"], info["d_before_strip"]
        ax.axvline(lx, color="purple", ls="--", label=f"dividing line x={lx:.2f}")
        ax.axvspan(lx - d, lx + d, color="orange", alpha=0.15,
                   label=f"strip (width 2d, d={d:.3f})")
        if info["strip"]:
            sx, sy = zip(*info["strip"])
            ax.scatter(sx, sy, s=22, facecolors="none", edgecolors="orange")

    if bf:
        ax.plot([bf[1][0], bf[2][0]], [bf[1][1], bf[2][1]], "r-", lw=4,
                alpha=0.5, label=f"Brute-Force  d={bf[0]:.6f}")
        ax.scatter(*zip(bf[1], bf[2]), c="red", s=90, zorder=5)
    if dc:
        ax.plot([dc[1][0], dc[2][0]], [dc[1][1], dc[2][1]], "b--", lw=2,
                label=f"Divide&Conquer d={dc[0]:.6f}")
        ax.scatter(*zip(dc[1], dc[2]), c="blue", s=40, zorder=6)

    ax.set_title("Closest pair of points")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_runtime(sizes=(100, 200, 400, 800, 1600, 3200, 6400),
                 fname="closest_pair_runtime.png"):
    tb, td = [], []
    print("\nPerformance (ms)      N    Brute-Force   Divide&Conquer   agree")
    for n in sizes:
        pts = gen_points(n)
        b, t1 = timed(brute_force, pts)
        d, t2 = timed(divide_and_conquer, pts)
        tb.append(t1); td.append(t2)
        print(f"{n:>22} {t1:>12.3f} {t2:>16.3f}   {abs(b[0]-d[0]) < 1e-9}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, tb, "o-r", label="Brute-Force  O(n²)")
    ax.plot(sizes, td, "s-b", label="Divide & Conquer  O(n log n)")
    ax.set_xlabel("N"); ax.set_ylabel("Running time (ms)")
    ax.set_title("Running time vs N"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ animation
def animate_dc(pts, fname="closest_pair_animation.gif", fps=1.5):
    """Animate every divide (split) and combine (merge) step of the
    Divide-and-Conquer algorithm and save it as a GIF."""
    trace = []
    divide_and_conquer(pts, None, trace)
    xs, ys = zip(*pts)
    padx = (max(xs) - min(xs)) * 0.05 + 1e-9
    pady = (max(ys) - min(ys)) * 0.05 + 1e-9
    fig, ax = plt.subplots(figsize=(8, 6.5))

    def draw(i):
        ax.clear()
        ev = trace[i]
        sub = ev["pts"]
        x0, x1 = min(p[0] for p in sub), max(p[0] for p in sub)
        ax.axvspan(x0, x1, color="skyblue", alpha=0.18, label="current sub-problem")
        ax.scatter(xs, ys, s=16, c="lightgray", zorder=2)
        ax.scatter(*zip(*sub), s=26, c="steelblue", zorder=3)

        if ev["kind"] == "split":
            ax.axvline(ev["line_x"], color="purple", ls="--", label="dividing line")
            title = f"DIVIDE: {len(sub)} points split at x = {ev['line_x']:.2f}"
        elif ev["kind"] == "base":
            d, p, q = ev["best"]
            ax.plot([p[0], q[0]], [p[1], q[1]], "g-", lw=3, zorder=4,
                    label=f"base-case best d={d:.3f}")
            ax.scatter(*zip(p, q), c="green", s=60, zorder=5)
            title = f"BASE CASE: brute force on {len(sub)} points"
        else:
            d, (_, p, q) = ev["d"], ev["best"]
            dnew = ev["best"][0]
            lx = ev["line_x"]
            ax.axvline(lx, color="purple", ls="--", label="dividing line")
            ax.axvspan(lx - d, lx + d, color="orange", alpha=0.25,
                       label=f"strip (2d), d={d:.3f}")
            if ev["strip"]:
                ax.scatter(*zip(*ev["strip"]), s=60, facecolors="none",
                           edgecolors="orange", linewidths=1.8, zorder=4)
            ax.plot([p[0], q[0]], [p[1], q[1]], "r-", lw=3, zorder=5,
                    label=f"best after merge d={dnew:.3f}")
            ax.scatter(*zip(p, q), c="red", s=60, zorder=6)
            tag = "strip improved it" if dnew < d - 1e-12 else "no better pair across strip"
            title = f"MERGE: {len(sub)} points, d {d:.3f} -> {dnew:.3f} ({tag})"

        ax.set_xlim(min(xs) - padx, max(xs) + padx)
        ax.set_ylim(min(ys) - pady, max(ys) + pady)
        ax.set_title(f"Step {i + 1}/{len(trace)}  {title}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=7)

    anim = FuncAnimation(fig, draw, frames=len(trace), repeat=True)
    anim.save(fname, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return len(trace)


# ----------------------------------------------------------------------- main
def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.readline()
    n = int(raw.strip())
    pts = gen_points(n) if n > 0 else []

    if n < 2:
        print("Brute-Force: N < 2, no pair exists")
        print("Divide-and-Conquer: N < 2, no pair exists")
        print("Time (ms): 0.000 0.000")
        return

    bf, t_bf = timed(brute_force, pts)
    info = {}
    dc, t_dc = timed(divide_and_conquer, pts, info)

    print(fmt(bf))
    print(fmt(dc))
    print(f"Brute-Force: {t_bf:.3f} ms, Divide-and-Conquer: {t_dc:.3f} ms")

    print("\nAgreement check:", "PASS" if abs(bf[0] - dc[0]) < 1e-9 else "FAIL")
    print("Corner-case tests:")
    run_tests()
    plot_points(pts, bf, dc, info)
    plot_runtime()
    # animation uses a small sample so every step stays readable
    sample = pts if n <= 40 else random.sample(pts, 40)
    frames = animate_dc(sample)
    print(f"\nSaved closest_pair_points.png, closest_pair_runtime.png and "
          f"closest_pair_animation.gif ({frames} frames, {len(sample)} points)")


if __name__ == "__main__":
    main()