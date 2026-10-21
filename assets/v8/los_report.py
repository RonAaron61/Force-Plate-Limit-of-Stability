"""PDF report for the LOS app (matplotlib only, no extra dependency). A4 portrait, text >= 9 pt.

build_report(path, a, meta, lt) -> path
  a    : parsed args of the session (los_live.make_parser())
  meta : dict(name, height_cm, when(datetime), upper, stance, footwear, version)
  lt   : finished los_live.LosTest (rows, paths, names, lean)
"""
import textwrap
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse, Circle
from matplotlib.backends.backend_pdf import PdfPages
import los_live as L

W, H = 8.27, 11.69
ACCENT, INK, MUTED, LINE, SOFT, BAND = "#1f4e79", "#1b2733", "#6b7785", "#d5dbe3", "#f4f6f9", "#dbe7f3"
PALETTE = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#17a2b8", "#c0392b", "#7f7f7f"]
SHORT = {"kanan": "R", "kanan-atas": "FR", "atas": "F", "kiri-atas": "FL", "kiri": "L", "kiri-bawah": "BL",
         "bawah": "B", "kanan-bawah": "BR"}


def _f(x, fmt):
    return "-" if x is None or x != x else fmt.format(x)


class _Page:
    def __init__(self, title, subtitle, when):
        self.fig = Figure(figsize=(W, H), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis("off")
        self.ax.add_patch(Rectangle((0, 0), W, 0.95, fc=ACCENT, ec="none"))
        self.ax.text(0.5, 0.40, title, color="white", fontsize=18, weight="bold", va="center")
        self.ax.text(0.5, 0.72, subtitle, color="#cfe0f1", fontsize=10, va="center")
        self.ax.text(W - 0.5, 0.40, when.strftime("%d %b %Y"), color="white", fontsize=11, ha="right", va="center")
        self.ax.text(W - 0.5, 0.72, when.strftime("%H:%M"), color="#cfe0f1", fontsize=10, ha="right", va="center")

    def footer(self, i, n, left):
        self.ax.plot([0.5, W - 0.5], [11.25, 11.25], color=LINE, lw=0.8)
        self.ax.text(0.5, 11.42, left, color=MUTED, fontsize=8, va="center")
        self.ax.text(W - 0.5, 11.42, f"Page {i}/{n}", color=MUTED, fontsize=8, ha="right", va="center")

    def card(self, x, y, w, h, title, items, key_w=1.35):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=SOFT, ec=LINE, lw=0.8))
        self.ax.text(x + 0.15, y + 0.25, title, color=ACCENT, fontsize=11, weight="bold", va="center")
        for k, (lab, val) in enumerate(items):
            yy = y + 0.58 + k * 0.235
            self.ax.text(x + 0.15, yy, lab, color=MUTED, fontsize=9, va="center")
            self.ax.text(x + 0.15 + key_w, yy, str(val), color=INK, fontsize=9.5, va="center")

    def table(self, x, y, widths, header, rows, row_h=0.27, fs=9.5):
        """rows: list of (kind, cells); kind 'block' (section header), 'data', 'mean'. Returns the y below the table."""
        ax, wt = self.ax, sum(widths)
        hh = row_h * 1.7 if any("\n" in h for h in header) else row_h
        ax.add_patch(Rectangle((x, y), wt, hh, fc=ACCENT, ec="none"))
        def put(cells, yy, hgt, color, weight, size):
            cx = x
            for j, (c, w) in enumerate(zip(cells, widths)):
                if j == 0: ax.text(cx + 0.08, yy + hgt / 2, c, color=color, fontsize=size, weight=weight, va="center", linespacing=1.15)
                else: ax.text(cx + w - 0.08, yy + hgt / 2, c, color=color, fontsize=size, weight=weight, va="center", ha="right", linespacing=1.15)
                cx += w
        put(header, y, hh, "white", "bold", fs - 0.5)
        y0 = y; y = y + hh - row_h
        for i, (kind, cells) in enumerate(rows):
            yy = y + row_h * (i + 1)
            if kind == "block":
                ax.add_patch(Rectangle((x, yy), wt, row_h, fc=BAND, ec="none"))
                ax.text(x + 0.08, yy + row_h / 2, cells[0], color=ACCENT, fontsize=fs, weight="bold", va="center")
                continue
            put(cells, yy, row_h, INK, "bold" if kind == "mean" else "normal", fs)
            ax.plot([x, x + wt], [yy + row_h, yy + row_h], color=LINE, lw=0.5)
        return y + row_h * (len(rows) + 1)


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) == r.get(key) and r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def _settings(a, meta, lt):
    lean = lt.lean
    subj = [("Name", meta["name"] or "-"), ("Height", f"{meta['height_cm']:g} cm" if meta.get("height_cm") else "-"),
            ("Date", meta["when"].strftime("%d %B %Y")), ("Time", meta["when"].strftime("%H:%M")),
            ("Upper extremities", meta["upper"]), ("Stance position", meta["stance"]), ("Footwear", meta["footwear"]),
            ("Trials completed", f"{len([r for r in lt.rows if r['outcome'] == 'ok'])}/{lt.total}  (ok / planned)")]
    test = [("Test", ("Lean" if lean else "Target") + f", {len(lt.names)} directions"),
            ("Repetitions", f"{a.reps} x per direction ({a.order})")]
    if lean: test.append(("Lean window", f"{a.lean_s:g} s after the light"))
    else: test.append(("Time limits", f"reach {a.reach_timeout:g} s, return {a.return_timeout:g} s"))
    test += [("Reference ellipse", f"{2 * a.los_x:g} x {2 * a.los_y:g} cm (placeholder)"),
             ("Light distance", f"{a.target_pct:g} % of the ellipse"),
             ("Light / home radius", f"{a.target_r:g} / {a.home_r:g} cm"),
             ("Onset threshold", f"{a.onset_cm:g} cm"),
             ("Sampling / low-pass", f"{a.fs:g} Hz / {a.cutoff:g} Hz")]
    return subj, test


def _summary_rows(a, lt):
    names = lt.names
    if lt.lean:
        hdr = ["Direction", "n ok", "Reaction\n(ms)", "Velocity\n(m/s)", "Excursion\n(m)", "Dir. control\n(%)"]
        wd = [1.6, 0.6, 1.1, 1.2, 1.3, 1.47]
        if a.height_m: hdr.append("Angle\n(deg)"); wd = [1.45, 0.55, 1.0, 1.05, 1.15, 1.2, 0.87]
        rows = []
        for d in names:
            rs = [r for r in lt.rows if r["direction"] == d]
            ok = [r for r in rs if r["outcome"] == "ok"]
            cells = [L.EN[d], str(len(ok)), _f(_mean(ok, "reaction_ms"), "{:.0f}"), _f(_mean(ok, "velocity_m_s"), "{:.3f}"),
                     _f(_mean(ok, "max_excursion_m"), "{:.3f}"), _f(_mean(ok, "direction_control_pct"), "{:.1f}")]
            if a.height_m: cells.append(_f(_mean(ok, "angle_deg"), "{:.2f}"))
            rows.append(("data", cells))
        return hdr, wd, rows
    hdr = ["Direction", "n ok", "Reaction\n(s)", "To target\n(s)", "Return\n(s)", "Cycle\n(s)", "Max exc.\n(%)"]
    wd = [1.6, 0.55, 1.0, 1.1, 1.0, 0.95, 1.07]
    rows = []
    for d in names:
        ok = [r for r in lt.rows if r["direction"] == d and r["outcome"] == "ok"]
        rows.append(("data", [L.EN[d], str(len(ok)), _f(_mean(ok, "reaction_s"), "{:.2f}"), _f(_mean(ok, "time_to_target_s"), "{:.2f}"),
                              _f(_mean(ok, "return_s"), "{:.2f}"), _f(_mean(ok, "cycle_s"), "{:.2f}"),
                              _f(_mean(ok, "max_excursion_pct"), "{:.0f}")]))
    return hdr, wd, rows


def _plots(pg, y, h, a, lt):
    fig = pg.fig
    def axes(x, w, **kw): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)
    ax = axes(0.75, 3.1)
    ax.set_title("CoP paths (screen view)", fontsize=10, color=ACCENT, weight="bold")
    ax.add_patch(Ellipse((0, 0), 2 * a.los_x, 2 * a.los_y, fill=False, ec="#d9534f", lw=1.4, alpha=.8))
    ax.add_patch(Circle((0, 0), a.home_r, fill=False, ls="--", ec="0.5", lw=1))
    for d in lt.names:
        tx, ty = lt.target_xy(d)
        ax.add_patch(Circle((tx, ty), a.target_r, fc="#eef0f3", ec="0.7", lw=.8, zorder=1))
    for k, (d, arr) in enumerate(lt.paths):
        if lt.lean: arr = arr[arr[:, 0] <= arr[0, 0] + a.lean_s]
        ax.plot(arr[:, 1], arr[:, 2], lw=1.1, alpha=.85, color=PALETTE[L.CANON.index(d) % len(PALETTE)], zorder=3)
    lim = max(a.los_x, a.los_y) * 1.55
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.tick_params(labelsize=8); ax.set_xlabel("Left  <-  ML (cm)  ->  Right", fontsize=8.5)
    ax.set_ylabel("Backward  <-  AP (cm)  ->  Forward", fontsize=8.5)
    pr = axes(4.75, 3.0, projection="polar")
    ang = [np.radians(L.DIRS[d]) for d in lt.names]
    if lt.lean:
        val = [_mean([r for r in lt.rows if r["direction"] == d and r["outcome"] == "ok"], "max_excursion_m") * 100 for d in lt.names]
        ttl, unit = "Mean maximum excursion (cm)", ""
    else:
        val = [_mean([r for r in lt.rows if r["direction"] == d and r["outcome"] == "ok"], "cycle_s") for d in lt.names]
        ttl = "Mean cycle time (s)"
    order = np.argsort(ang); th = np.array(ang)[order]; vv = np.nan_to_num(np.array(val, float)[order])
    pr.plot(np.r_[th, th[0]], np.r_[vv, vv[0]], "-o", color=ACCENT, lw=1.6, ms=4)
    pr.fill(np.r_[th, th[0]], np.r_[vv, vv[0]], color=ACCENT, alpha=.15)
    pr.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    pr.set_xticklabels(["R", "FR", "F", "FL", "L", "BL", "B", "BR"], fontsize=8.5)
    pr.tick_params(axis="y", labelsize=7.5); pr.set_title(ttl, fontsize=10, color=ACCENT, weight="bold", pad=14)
    pr.set_ylim(0, max(1e-6, vv.max()) * 1.2)


def _notes(a, lt):
    if lt.lean:
        t = ("Reaction time: first moment the CoP is >= %.1f cm away from its position at light onset; it depends on this threshold and includes "
             "the low-pass filter delay (about 40 ms at 10 Hz) and display latency. Movement velocity: excursion gained between onset and the first "
             "time 95%% of the maximum is reached, divided by that time. Maximum CoP excursion: furthest CoP position along the direction of the lit "
             "light, measured from the CoP position at light onset. Direction control: (on-axis path - off-axis path) / on-axis path x 100 during the "
             "movement. Maximal COP angle (only if height was entered): atan(excursion / (0.55 x height)), assuming the centre of gravity at 55%% of "
             "height (not validated). Directions follow the screen: Forward = up. " % a.onset_cm)
    else:
        t = ("Reaction: first moment the CoP is >= %.1f cm from its start position after the target appears. To target: target onset until the CoP "
             "stayed inside the target circle for %.2f s. Return: from reaching the target until the CoP stayed inside the home circle for %.2f s. "
             "Max exc.: furthest projection toward the target as %% of the target distance. " % (a.onset_cm, a.dwell_s, a.dwell_s))
    t += ("These definitions are this software's own. They have not been validated against other systems, reference data or norms, and the %g x %g cm "
          "reference ellipse is a placeholder. Trials not marked 'ok' are excluded from the means." % (2 * a.los_x, 2 * a.los_y))
    return textwrap.wrap(t, 100)


def _detail_pages(a, lt, when, sub):
    pages, names = [], lt.names
    if lt.lean:
        mets = [("Reaction time (ms)", "reaction_ms", "{:.0f}"), ("Movement velocity (m/s)", "velocity_m_s", "{:.3f}"),
                ("Maximum COP excursion (m)", "max_excursion_m", "{:.3f}"), ("Direction control (%)", "direction_control_pct", "{:.1f}")]
        if a.height_m: mets.append(("Maximal COP angle (deg)", "angle_deg", "{:.2f}"))
        n = max(len([r for r in lt.rows if r["direction"] == d]) for d in names)
        lab_w, mean_w = 2.35, 0.85
        cw = min(0.95, (W - 1.0 - lab_w - mean_w) / max(n, 1))
        widths = [lab_w] + [cw] * n + [mean_w]
        hdr = [""] + [f"T{k + 1}" for k in range(n)] + ["Mean"]
        per_page = max(1, int((H - 2.7) / (0.27 * (len(mets) + 1))))
        for p0 in range(0, len(names), per_page):
            pg = _Page("Detailed results", sub, when); rows = []
            for d in names[p0:p0 + per_page]:
                rs = [r for r in lt.rows if r["direction"] == d]
                rows.append(("block", [L.EN[d]]))
                for lab, key, fmt in mets:
                    vals = [_f(r[key], fmt) + ("" if r["outcome"] == "ok" else "*") for r in rs]
                    ok = [r for r in rs if r["outcome"] == "ok"]
                    rows.append(("data", [lab] + vals + [""] * (n - len(vals)) + [_f(_mean(ok, key), fmt)]))
            y = pg.table(0.5, 1.3, widths, hdr, rows, row_h=0.27, fs=9.5 if n <= 8 else 8.5); pg.y_end = y
            if any(r["outcome"] != "ok" for r in lt.rows):
                pg.ax.text(0.5, y + 0.25, "* trial not counted (outcome other than 'ok'); see the trials CSV for the reason.", fontsize=8.5, color=MUTED)
            pages.append(pg)
    else:
        hdr = ["#", "Direction", "Reaction\n(s)", "To target\n(s)", "Return\n(s)", "Cycle\n(s)", "Max exc.\n(%)", "Path eff.", "Outcome"]
        widths = [0.35, 1.25, 0.9, 0.95, 0.85, 0.8, 0.95, 0.75, 1.0]
        per_page = int((H - 2.7) / 0.27)
        for p0 in range(0, len(lt.rows), per_page):
            pg = _Page("Detailed results", sub, when)
            rows = [("data", [str(r["trial"]), L.EN[r["direction"]], _f(r["reaction_s"], "{:.2f}"), _f(r["time_to_target_s"], "{:.2f}"),
                              _f(r["return_s"], "{:.2f}"), _f(r["cycle_s"], "{:.2f}"), _f(r["max_excursion_pct"], "{:.0f}"),
                              _f(r["path_efficiency"], "{:.2f}"), r["outcome"]]) for r in lt.rows[p0:p0 + per_page]]
            pg.y_end = pg.table(0.5, 1.3, widths, hdr, rows, row_h=0.27, fs=9)
            pages.append(pg)
    return pages


def draw_session(fig, box, a, lt, S, fs=8.5, layout="stack"):
    """Whole-recording CoP: path coloured by time + ML(t) and AP(t) with the trial windows shaded.
    S = dict(t (s), sx, sy (cm, screen view: +x right, +y forward, home = 0)).  box = (left, bottom, w, h) in figure fractions."""
    from matplotlib.collections import LineCollection
    Wi, Hi = fig.get_size_inches(); l, b, w, h = box; bw, bh = w * Wi, h * Hi
    def ax_at(x, y, ww, hh, **kw): return fig.add_axes([l + x / Wi, b + y / Hi, ww / Wi, hh / Hi], **kw)
    if layout == "side":        # path on the left, time series stacked on the right (wide window)
        side = min(bh * 0.78, bw * 0.40); y0 = (bh - side) / 2
        ax = ax_at(0.8, y0, side, side)
    else:                       # A4 page: path on top, time series below
        side = min(bw * 0.72, bh * 0.50); y0 = bh - side - 0.35
        ax = ax_at((bw - side) / 2 - 0.1, y0, side, side)
    t, sx, sy = S["t"], S["sx"], S["sy"]
    ax.add_patch(Ellipse((0, 0), 2 * a.los_x, 2 * a.los_y, fc="#fdecea", ec="#d9534f", lw=1.4, alpha=.7, zorder=1))
    ax.add_patch(Circle((0, 0), a.home_r, fill=False, ls="--", ec="0.5", lw=1, zorder=2))
    if lt.lean:
        for d in lt.names:
            tx, ty = lt.target_xy(d); ax.add_patch(Circle((tx, ty), a.target_r, fc="#eef0f3", ec="0.7", lw=.8, zorder=1))
    ok = np.isfinite(sx) & np.isfinite(sy)
    pts = np.c_[sx, sy]; seg = np.stack([pts[:-1], pts[1:]], axis=1); good = ok[:-1] & ok[1:]
    lc = LineCollection(seg[good], cmap="viridis", linewidths=1.0, zorder=3); lc.set_array(t[:-1][good]); ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.03); cb.set_label("time (s)", fontsize=fs); cb.ax.tick_params(labelsize=fs - 1)
    lim = max(a.los_x, a.los_y, np.nanmax(np.abs(np.r_[sx[ok], sy[ok]])) if ok.any() else 1) * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=.25); ax.tick_params(labelsize=fs)
    ax.set_title("CoP path, whole recording", fontsize=fs + 1.5, color=ACCENT, weight="bold")
    ax.set_xlabel("Left  <-  ML (cm)  ->  Right", fontsize=fs); ax.set_ylabel("Backward  <-  AP (cm)  ->  Forward", fontsize=fs)
    if layout == "side":
        x0 = 0.8 + side + 1.7; th = (bh - 1.9) / 2
        ax_ap = ax_at(x0, 0.6, bw - x0 - 0.2, th); ax_ml = ax_at(x0, 0.6 + th + 0.7, bw - x0 - 0.2, th, sharex=ax_ap)
    else:
        th = (y0 - 1.9) / 2
        ax_ap = ax_at(0.75, 0.55, bw - 1.0, th); ax_ml = ax_at(0.75, 0.55 + th + 0.45, bw - 1.0, th, sharex=ax_ap)
    for axx, v, lab in ((ax_ml, sx, "ML (cm)  + = right"), (ax_ap, sy, "AP (cm)  + = forward")):
        axx.plot(t, v, color="#1b2733", lw=0.8, zorder=3); axx.axhline(0, color="0.6", lw=.6)
        axx.set_ylabel(lab, fontsize=fs); axx.tick_params(labelsize=fs); axx.grid(alpha=.25)
    ax_ap.set_xlabel("time (s)", fontsize=fs)
    plt_top = ax_ml.get_ylim()[1]
    for r in lt.rows:
        t0 = r.get("t_cue_rec_s")
        if t0 is None or t0 != t0: continue
        dur = a.lean_s if lt.lean else (r["cycle_s"] if r.get("cycle_s") == r.get("cycle_s") else a.reach_timeout)
        col = PALETTE[L.CANON.index(r["direction"]) % len(PALETTE)]
        for axx in (ax_ml, ax_ap): axx.axvspan(t0, t0 + dur, color=col, alpha=.16, lw=0)
        ax_ml.text(t0 + dur / 2, plt_top, SHORT.get(r["direction"], ""), fontsize=fs - 1, ha="center", va="bottom", color=col, weight="bold", clip_on=False)
    ax_ml.set_title("CoP over time" if layout == "side" else "CoP over time (shaded = trial window, letters = light direction)", fontsize=fs + 0.5, color=ACCENT, weight="bold", pad=14)


def build_report(path, a, meta, lt, session=None):
    when = meta["when"]
    sub = (("Lean test" if lt.lean else "Target test") + f" - {len(lt.names)} directions - " + (meta["name"] or "no name"))
    first = _Page("Limits of Stability Report", sub, when)
    subj, test = _settings(a, meta, lt)
    first.card(0.5, 1.2, 3.55, 2.62, "Subject", subj)
    first.card(4.2, 1.2, 3.57, 2.62, "Test settings", test, key_w=1.45)
    first.ax.text(0.5, 4.2, "Mean results by direction", fontsize=12, weight="bold", color=ACCENT, va="center")
    hdr, wd, rows = _summary_rows(a, lt)
    y = first.table(0.5, 4.4, wd, hdr, rows, row_h=0.28, fs=9.5)
    h_plot = min(3.2, 11.0 - (y + 0.75))
    if lt.paths and h_plot > 2.0: _plots(first, y + 0.75, h_plot, a, lt)
    pages = [first]
    if session is not None and len(session["t"]) > 10:
        pg = _Page("CoP recording", sub, when); draw_session(pg.fig, (0.04, 0.055, 0.93, 0.835), a, lt, session); pages.append(pg)
    pages += _detail_pages(a, lt, when, sub)
    lines = _notes(a, lt)
    need = 0.5 + 0.18 * len(lines)
    if pages[-1] is first or pages[-1].y_end + 0.35 + need > 11.1:
        pages.append(_Page("Definitions and limitations", sub, when)); pages[-1].y_end = 0.9
    last = pages[-1]; ny = last.y_end + 0.5
    last.ax.text(0.5, ny, "Definitions and limitations", fontsize=11, weight="bold", color=ACCENT, va="center"); ny += 0.3
    for ln in lines: last.ax.text(0.5, ny, ln, fontsize=9, color=INK, va="center"); ny += 0.18
    left = f"{meta.get('version', 'LOS app')}  |  {meta['name'] or '-'}  |  {when:%Y-%m-%d %H:%M}"
    with PdfPages(path, metadata={"Title": "LOS report - " + (meta["name"] or ""), "Creator": meta.get("version", "LOS app")}) as pdf:
        for i, pg in enumerate(pages, 1):
            pg.footer(i, len(pages), left); pdf.savefig(pg.fig)
    return path
