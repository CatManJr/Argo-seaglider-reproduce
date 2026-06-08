"""
Nitrate depth-time sections — 3 stacked panels (16:9).

  a. SOGOS float observed nitrate — 6 profiles / ~30 days  (paper Fig. 7a)
  b. SG660 Seaglider RFR nitrate — same period              (paper Fig. 7b)
  c. SG659 Seaglider RFR nitrate — same period              (extension)
"""
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = Path(__file__).parent / "data" / "zenodo_main"
OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

FIG_W, FIG_H = 12, 16
TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 14
CBAR_LABEL_FONTSIZE = 14
SUBPLOT_HSPACE = 0.14
N_VMIN, N_VMAX = 24.0, 36.0
GLIDER_DP = 1  # L3 Seaglider product: 1-m depth intervals (Appendix a)
ARGO_SHALLOW = 400
FLOAT_DEEP_HALF = 15  # spot-sample footprint [dbar]; 50-dbar spacing leaves gaps use +-15 dbar bucket

WINDOW_START = datetime(2019, 5, 11)
WINDOW_END = datetime(2019, 6, 10, 23, 59, 59)
FLOAT_PROFILE_START = datetime(2019, 5, 16)
FLOAT_PROFILE_END = datetime(2019, 6, 10, 23, 59, 59)
N_FLOAT_PROFILES = 6


def nitrate_cmap():
    colors = ["#ffffff", "#8ecf8e", "#2a6db5", "#000000"]
    cmap = LinearSegmentedColormap.from_list("nitrate_wgbk", colors, N=256)
    cmap.set_bad("#ffffff")
    return cmap


def load_glider(tag):
    fp = DATA_DIR / f"sogos_glider{tag}_RFRpred_setB_modG.csv"
    return pd.read_csv(fp, parse_dates=["juld"])


def load_float():
    fp = DATA_DIR / "sogos_float5906030_RFRtest_setB_modG.csv"
    df = pd.read_csv(fp)
    base = datetime(2019, 1, 1)
    df["juld"] = df["yearday"].apply(lambda yd: base + timedelta(days=float(yd)))
    return df


def pressure_edges(pres_max, dp):
    top = int(np.ceil(pres_max / dp) * dp)
    return np.arange(0, top + dp, dp)


def daily_time_edges(t0, t1):
    n0 = mdates.date2num(t0)
    n1 = mdates.date2num(t1)
    return np.arange(n0, n1 + 1, 1.0)


def profile_time_edges(t_centers):
    t_centers = np.asarray(t_centers, dtype=float)
    if len(t_centers) == 0:
        return t_centers
    if len(t_centers) == 1:
        half = 0.5 / 24
        return np.array([t_centers[0] - half, t_centers[0] + half])
    mid = 0.5 * (t_centers[1:] + t_centers[:-1])
    left = t_centers[0] - (mid[0] - t_centers[0])
    right = t_centers[-1] + (t_centers[-1] - mid[-1])
    return np.concatenate([[left], mid, [right]])


def select_sogos_float_profiles(float_df):
    meta = float_df.groupby("profid")["juld"].first().sort_values()
    in_window = meta[(meta >= FLOAT_PROFILE_START) & (meta <= FLOAT_PROFILE_END)]
    if len(in_window) < N_FLOAT_PROFILES:
        raise ValueError(f"Expected {N_FLOAT_PROFILES} float profiles, found {len(in_window)}")
    keep = in_window.index[:N_FLOAT_PROFILES]
    return float_df[float_df["profid"].isin(keep)]


def clip_time_window(df, t0=WINDOW_START, t1=WINDOW_END):
    return df[(df["juld"] >= t0) & (df["juld"] <= t1)].copy()


def fill_shallow_column(grid, ti, shallow, var, p_edges):
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    shallow_mask = p_centers <= ARGO_SHALLOW
    if len(shallow) >= 2:
        vals = np.interp(p_centers[shallow_mask], shallow["pressure"], shallow[var])
        grid[shallow_mask, ti] = vals


def fill_deep_spots(grid, ti, deep, var, p_edges):
    """
    Argo biogeochemical spot samples below 400 dbar (~50 dbar apart).
    Paint only a small block at each native depth — gaps stay white.
    """
    if deep.empty:
        return
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    for _, row in deep.iterrows():
        p_val, v_val = row["pressure"], row[var]
        mask = (p_centers >= p_val - FLOAT_DEEP_HALF) & (p_centers < p_val + FLOAT_DEEP_HALF)
        grid[mask, ti] = v_val


def fill_profile_column(grid, ti, grp, var, p_edges):
    sub = grp[["pressure", var]].dropna().sort_values("pressure")
    if sub.empty:
        return
    shallow = sub[sub["pressure"] <= ARGO_SHALLOW]
    deep = sub[sub["pressure"] > ARGO_SHALLOW]
    fill_shallow_column(grid, ti, shallow, var, p_edges)
    fill_deep_spots(grid, ti, deep, var, p_edges)


def make_float_section(float_df, var, p_edges, t_edges):
    """Daily time grid; only the 6 observation days are filled."""
    n_p = len(p_edges) - 1
    n_t = len(t_edges) - 1
    grid = np.full((n_p, n_t), np.nan)

    profiles = sorted(
        ((pid, grp["juld"].iloc[0], grp) for pid, grp in float_df.groupby("profid", sort=False)),
        key=lambda item: item[1],
    )

    t0 = t_edges[0]
    for _, t_prof, grp in profiles:
        ti = int(np.floor(mdates.date2num(t_prof) - t0))
        if 0 <= ti < n_t:
            fill_profile_column(grid, ti, grp, var, p_edges)

    return t_edges, p_edges, grid


def assign_l3_profile_column(grid, ti, pressures, values, n_p):
    """
    Place L3 ~1-m profile onto integer 1-dbar levels within the observed
    depth span only (Appendix vertical interpolation, not plot extrapolation).
    """
    p_lo = int(np.floor(pressures.min()))
    p_hi = min(int(np.ceil(pressures.max())), n_p - 1)
    if p_hi <= p_lo:
        return
    levels = np.arange(p_lo, p_hi + 1)
    grid[levels, ti] = np.interp(levels, pressures, values)


def make_glider_section(df, var, p_edges):
    """
    Paper Fig. 7b: L3 1-m data + RFR, one column per dive.
    Columns are filled on integer 1-dbar levels within each profile span.
    """
    n_p = len(p_edges) - 1
    p_max = p_edges[-1]

    profiles = sorted(
        ((pid, grp["juld"].iloc[0], grp) for pid, grp in df.groupby("profid", sort=False)),
        key=lambda item: item[1],
    )
    n_t = len(profiles)
    grid = np.full((n_p, n_t), np.nan)
    t_centers = np.empty(n_t)

    for ti, (_, t0, grp) in enumerate(profiles):
        t_centers[ti] = mdates.date2num(t0)
        ok = (grp["pressure"] >= 0) & (grp["pressure"] <= p_max) & np.isfinite(grp[var])
        sub = grp.loc[ok, ["pressure", var]].sort_values("pressure")
        if len(sub) < 2:
            continue
        assign_l3_profile_column(
            grid, ti, sub["pressure"].values, sub[var].values, n_p
        )

    return profile_time_edges(t_centers), p_edges, grid


def plot_section(ax, t_edges, p_edges, grid, *, cmap, vmin, vmax):
    t, p = np.meshgrid(t_edges, p_edges)
    return ax.pcolormesh(
        t, p, grid, cmap=cmap, shading="flat", vmin=vmin, vmax=vmax, rasterized=True
    )


def ten_day_ticks(t_lim):
    start = mdates.num2date(t_lim[0]).replace(hour=0, minute=0, second=0, microsecond=0)
    end = mdates.num2date(t_lim[1])
    return mdates.date2num(pd.date_range(start, end, freq="10D").to_pydatetime())


def format_axes(axes, t_lim, p_lim, pres_ticks, date_ticks):
    date_fmt = mdates.DateFormatter("%m-%d")
    for ax in axes:
        ax.set_xlim(t_lim)
        ax.set_ylim(p_lim)
        ax.invert_yaxis()
        ax.set_yticks(pres_ticks)
        ax.set_xticks(date_ticks)
        ax.xaxis.set_major_formatter(date_fmt)
        ax.set_ylabel("Pressure [dbar]", rotation=90, labelpad=10, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(
            axis="both", direction="in", top=False, right=False, labelsize=TICK_FONTSIZE
        )

    axes[-1].set_xlabel("Date (2019)", fontsize=AXIS_LABEL_FONTSIZE)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def panel_label(ax, letter):
    ax.text(
        0.02, 0.98, f"{letter}.",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold", va="top", ha="left",
    )


def main():
    d659 = clip_time_window(load_glider(659))
    d660 = clip_time_window(load_glider(660))
    flt = select_sogos_float_profiles(load_float())

    pres_max = max(d659["pressure"].max(), d660["pressure"].max())
    p_edges = pressure_edges(pres_max, GLIDER_DP)
    p_lim = (0, p_edges[-1])
    pres_ticks = np.arange(0, p_edges[-1] + 1, 200)

    t_lim = (mdates.date2num(WINDOW_START), mdates.date2num(WINDOW_END))
    t_daily = daily_time_edges(WINDOW_START, WINDOW_END)
    date_ticks = ten_day_ticks(t_lim)

    n660 = d660["profid"].nunique()
    n659 = d659["profid"].nunique()

    cmap = nitrate_cmap()
    sections = [
        ("a", "SOGOS float  (observed)", make_float_section(flt, "nitrate", p_edges, t_daily)),
        ("b", f"SG660  Seaglider RFR  ({n660} profiles)", make_glider_section(d660, "nitrate_G", p_edges)),
        ("c", f"SG659  Seaglider RFR  ({n659} profiles)", make_glider_section(d659, "nitrate_G", p_edges)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)

    im = None
    for ax, (letter, title, (te, pe, gr)) in zip(axes, sections):
        im = plot_section(ax, te, pe, gr, cmap=cmap, vmin=N_VMIN, vmax=N_VMAX)
        panel_label(ax, letter)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, loc="left", pad=6)

    format_axes(axes, t_lim, p_lim, pres_ticks, date_ticks)

    cax = fig.add_axes([0.92, 0.35, 0.015, 0.30])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Nitrate [µmol kg$^{-1}$]", fontsize=CBAR_LABEL_FONTSIZE)
    cbar.set_ticks(np.arange(24, 37, 2))
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.97, bottom=0.08, hspace=SUBPLOT_HSPACE)

    out = OUTPUT / "fig7_nitrate_section.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")
    print(f"  float profiles: {N_FLOAT_PROFILES}")
    print(f"  SG660 profiles: {n660}  (paper reports 428)")
    print(f"  SG659 profiles: {n659}")


if __name__ == "__main__":
    main()
