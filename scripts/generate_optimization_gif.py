"""Generate the optimization animation GIF used in phd/03_learning.qmd.

The animation shows an iterative "learning" process where we try a sequence
of carefully chosen (intercept, slope) parameter pairs for the shoe-size
vs. age model, and watch the sum of squares drop until we reach the best
pair (18.5, 1.5).

The data is the same as in phd/02_whats_a_model.qmd (same rng seed).

Output: images/optimization.gif
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_PATH = ROOT / "images" / "optimization.gif"


def build_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    ages_a = rng.integers(1, 11, size=10)
    sizes_a = 19 + 1.6 * ages_a + rng.normal(0, 0.7, size=10)
    ages_b = rng.integers(1, 11, size=10)
    sizes_b = 18 + 1.5 * ages_b + rng.normal(0, 0.7, size=10)
    return np.concatenate([ages_a, ages_b]), np.concatenate([sizes_a, sizes_b])


def sum_of_squares(intercept: float, slope: float, ages: np.ndarray, sizes: np.ndarray) -> float:
    predicted = intercept + slope * ages
    return float(np.sum((sizes - predicted) ** 2))


def render_frame(
    k: int,
    show_label: bool,
    pairs: list[tuple[float, float]],
    ss_values: list[float],
    colors: list[str],
    ages: np.ndarray,
    sizes: np.ndarray,
    y_min_right: float,
    y_max_right: float,
) -> Image.Image:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=110)

    xs = list(range(1, k + 2))
    ys = ss_values[: k + 1]
    cs = colors[: k + 1]

    if len(xs) > 1:
        ax_left.plot(xs, ys, color="#888888", linewidth=1.5, zorder=2)
    for x, y, c in zip(xs, ys, cs):
        ax_left.scatter([x], [y], color=c, s=90, zorder=3, edgecolor="white", linewidth=0.8)

    if show_label:
        p = pairs[k]
        is_last = k == len(pairs) - 1
        ax_left.annotate(
            f"({p[0]}, {p[1]})",
            xy=(xs[-1], ys[-1]),
            xytext=(-28 if is_last else 0, 14),
            textcoords="offset points",
            color=cs[-1],
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor=cs[-1], boxstyle="round,pad=0.25", alpha=0.95),
        )

    ax_left.set_xlabel("Parameter pair index")
    ax_left.set_ylabel("Sum of squares")
    ax_left.set_xlim(0.3, len(pairs) + 0.7)
    ax_left.set_ylim(0, max(ss_values) * 1.28)
    ax_left.set_xticks(range(1, len(pairs) + 1))
    ax_left.grid(True, alpha=0.25)

    ax_right.scatter(ages, sizes, color="black", zorder=3)
    intercept, slope = pairs[k]
    line_x = np.array([1, 10])
    line_y = intercept + slope * line_x
    ax_right.plot(line_x, line_y, color=cs[-1], linewidth=3, zorder=2)

    if show_label:
        p = pairs[k]
        ax_right.text(
            0.97,
            0.05,
            f"({p[0]}, {p[1]})",
            transform=ax_right.transAxes,
            color=cs[-1],
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", edgecolor=cs[-1], boxstyle="round,pad=0.3", alpha=0.9),
        )

    ax_right.set_xlabel("Age (years)")
    ax_right.set_ylabel("Shoe size (EU)")
    ax_right.set_xticks(range(1, 11))
    ax_right.set_xlim(0.5, 10.5)
    ax_right.set_ylim(y_min_right, y_max_right)
    ax_right.grid(True, alpha=0.25)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).copy().convert("P", palette=Image.ADAPTIVE, colors=192)
    plt.close(fig)
    return img


def main() -> None:
    ages, sizes = build_data()

    pairs: list[tuple[float, float]] = [
        (15.0, 2.5),
        (16.0, 2.2),
        (17.0, 2.0),
        (18.0, 1.7),
        (18.3, 1.6),
        (18.5, 1.5),
    ]
    ss_values = [sum_of_squares(a, b, ages, sizes) for a, b in pairs]
    print("Sum of squares per pair:", [round(v, 2) for v in ss_values])
    for i in range(1, len(ss_values)):
        assert ss_values[i] < ss_values[i - 1], (
            f"SS not monotonically decreasing at index {i}: {ss_values}"
        )

    colors = [
        "#4C72B0",
        "#8172B2",
        "#CCB974",
        "#DD8452",
        "#EE854A",
        "#D62728",
    ]

    y_min_right = float(min(sizes.min(), min(p[0] + p[1] * 1 for p in pairs))) - 1.0
    y_max_right = float(max(sizes.max(), max(p[0] + p[1] * 10 for p in pairs))) + 1.0

    frames: list[Image.Image] = []
    durations: list[int] = []

    for k in range(len(pairs)):
        frames.append(
            render_frame(k, True, pairs, ss_values, colors, ages, sizes, y_min_right, y_max_right)
        )
        durations.append(600)
        frames.append(
            render_frame(k, False, pairs, ss_values, colors, ages, sizes, y_min_right, y_max_right)
        )
        durations.append(150)

    frames.append(
        render_frame(
            len(pairs) - 1, True, pairs, ss_values, colors, ages, sizes, y_min_right, y_max_right
        )
    )
    durations.append(1800)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
