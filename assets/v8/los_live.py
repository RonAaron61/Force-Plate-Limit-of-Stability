#!/usr/bin/env python3
"""
Live CoP display for Limits of Stability (LOS) using a bilateral 4-load-cell force plate (v4).

Geometry and scaling were reverse-engineered from a Templo export (DataDiagramBalance.txt):
  - each plate: 4 load cells at corners (+-0.115 m in x, +-0.25 m in y)
  - left plate centred at x = 0, right plate centred at x = -0.353 m (Templo frame)
  - raw column x 500 = force in N (assumed; verify with a known weight)

Workflow (mirrors Templo): PREVIEW runs from program start -> plates EMPTY -> tare ->
patient steps on -> Start measurement -> Stop (saves a CSV).

Usage:
  python los_live.py --replay DataDiagramBalance.txt --fs 1000      # test without hardware
  python los_live.py --ni "Dev1/ai0:7" --fs 1000                    # live from NI DAQ
Lean test (4 lights: left/right/up/down, lean toward the lit one as far as you can; report table at the end):
  python los_live.py --ni "Dev1/ai0:7" --fs 1000 --los-test --mode lean --reps 3 --height-m 1.65
Settings form (respondent name, 4D/8D, all parameters; remembered between runs): python los_live.py   (no arguments)
8 directions: add --n-dirs 8.   Output: <outdir>/<respondent>/<respondent>_<time>_{raw,trials,summary,params}.*
On-screen buttons or keys (click the window first so it has focus):
  t = tare (plates EMPTY)   space = start/stop measurement   c = set centre (patient standing still)
  a = toggle autoscale   + / - = zoom   x / y = flip left-right / front-back   r = reset   q = quit
"""
import argparse, collections, csv, datetime, threading, time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

# ---------------- plate geometry (metres, Templo frame) ----------------
HX, HY = 0.115, 0.250
CELL_SIGN = np.array([(+1, -1), (+1, +1), (-1, -1), (-1, +1)], float)  # cells 1..4 -> (sx, sy)
PLATE_CENTRE_X = {"L": 0.0, "R": -0.353}
X_MIN, X_MAX = PLATE_CENTRE_X["R"] - HX, PLATE_CENTRE_X["L"] + HX       # CoP can never leave this box
N_PER_VOLT = 500.0          # raw (V) -> N, inferred from the export; VALIDATE
MIN_FZ = 50.0               # N; below this CoP is undefined. Tune to ~10-20 % of body weight.
ENVELOPE_MARGIN = 0.02      # m of slack around the plates before a sample is rejected


def cop_from_cells(F, min_fz=MIN_FZ, margin=ENVELOPE_MARGIN):
    """F: (n, 8) = L1..L4, R1..R4 in N. Returns dict: Fz, L_Fz, R_Fz, copx, copy (m), dropped (bool).
    CoP is NaN when total Fz < min_fz or when it falls outside the physical plate area."""
    out, Fz, mx, my = {}, 0, 0, 0
    for name, sl in (("L", slice(0, 4)), ("R", slice(4, 8))):
        f = F[:, sl]
        fz = f.sum(1)
        mx = mx + PLATE_CENTRE_X[name] * fz + (f * CELL_SIGN[:, 0] * HX).sum(1)   # force-weighted x
        my = my + (f * CELL_SIGN[:, 1] * HY).sum(1)
        Fz = Fz + fz
        out[name + "_Fz"] = fz
    ok = Fz > min_fz
    safe = np.where(ok, Fz, 1)
    cx = np.where(ok, mx / safe, np.nan)
    cy = np.where(ok, my / safe, np.nan)
    inside = (cx >= X_MIN - margin) & (cx <= X_MAX + margin) & (np.abs(cy) <= HY + margin)
    out["dropped"] = ok & ~inside
    out["Fz"] = Fz
    out["copx"] = np.where(inside, cx, np.nan)
    out["copy"] = np.where(inside, cy, np.nan)
    return out


# ---------------- data sources ----------------
class Source:
    def start(self): ...
    def read(self):  # -> (n, 8) newly available samples in N (may be empty)
        raise NotImplementedError
    def stop(self): ...


class ReplaySource(Source):
    """Plays a Templo txt export back in real time (decimal comma, tab separated)."""
    def __init__(self, path, fs, use_filtered=False):
        import pandas as pd
        df = pd.read_csv(path, sep="\t", decimal=",")
        if use_filtered:
            cols = [f"r{p}_Fz{i}_filtered" for p in "LR" for i in range(1, 5)]
            self.data = df[cols].to_numpy(float)
        else:
            cols = [f"r{p}_Fz{i}" for p in "LR" for i in range(1, 5)]
            self.data = df[cols].to_numpy(float) * N_PER_VOLT
        self.fs, self.i, self.t0 = fs, 0, None

    def start(self): self.t0 = time.perf_counter()

    def read(self):
        n_due = int((time.perf_counter() - self.t0) * self.fs)
        n_due = min(n_due, len(self.data))
        out = self.data[self.i:n_due]
        self.i = n_due
        if self.i >= len(self.data):  # loop
            self.i, self.t0 = 0, time.perf_counter()
        return out


class NISource(Source):
    """Continuous NI-DAQmx acquisition. Channel order must be L1,L2,L3,L4,R1,R2,R3,R4."""
    def __init__(self, channels, fs, vmin=-10.0, vmax=10.0, terminal="rse"):
        self.channels, self.fs, self.vmin, self.vmax, self.terminal = channels, fs, vmin, vmax, terminal
        self.buf, self.lock, self.run = [], threading.Lock(), False

    def start(self):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        tc = getattr(TerminalConfiguration, {"rse": "RSE", "nrse": "NRSE", "diff": "DIFF",
                                             "default": "DEFAULT"}[self.terminal])
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(self.channels, terminal_config=tc,
                                                  min_val=self.vmin, max_val=self.vmax)
        self.task.timing.cfg_samp_clk_timing(self.fs, sample_mode=AcquisitionType.CONTINUOUS,
                                             samps_per_chan=int(self.fs * 10))
        self.task.start()
        self.run = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        n = max(1, int(self.fs // 100))  # ~10 ms blocks
        while self.run:
            d = np.asarray(self.task.read(number_of_samples_per_channel=n, timeout=2.0)).T  # (n, 8) volts
            with self.lock:
                self.buf.append(d * N_PER_VOLT)

    def read(self):
        with self.lock:
            out, self.buf = self.buf, []
        return np.vstack(out) if out else np.empty((0, 8))

    def stop(self):
        self.run = False
        time.sleep(0.05)
        self.task.stop(); self.task.close()


# ---------------- LOS target test (state machine on the DAQ sample clock) ----------------
DIRS = {"kanan": 0, "kanan-atas": 45, "atas": 90, "kiri-atas": 135,
        "kiri": 180, "kiri-bawah": 225, "bawah": 270, "kanan-bawah": 315}   # screen frame, patient faces the screen
EN = {"kanan": "Right", "kanan-atas": "Forward-Right", "atas": "Forward", "kiri-atas": "Forward-Left",
      "kiri": "Left", "kiri-bawah": "Backward-Left", "bawah": "Backward", "kanan-bawah": "Backward-Right"}
CANON = ["atas", "bawah", "kanan", "kiri", "kanan-atas", "kiri-atas", "kanan-bawah", "kiri-bawah"]
LEAN_COLS = ["trial", "subject", "direction", "direction_en", "t_cue_s", "t_cue_rec_s", "reaction_ms", "velocity_m_s", "peak_speed_m_s",
             "max_excursion_m", "max_radial_m", "direction_control_pct", "angle_deg", "movement_s",
             "false_starts", "outcome", "onset_thr_cm", "lean_s"]
LEAN_METRICS = [("Reaction time (ms)", "reaction_ms", "{:.0f}"),
                ("Movement velocity (m/s)", "velocity_m_s", "{:.3f}"),
                ("Maximum COP excursion (m)", "max_excursion_m", "{:.3f}"),
                ("Direction control (%)", "direction_control_pct", "{:.1f}")]
TRIAL_COLS = ["trial", "subject", "direction", "target_x_cm", "target_y_cm", "t_target_on_s", "t_cue_rec_s", "reaction_s", "time_to_target_s",
              "movement_s", "return_s", "cycle_s", "max_excursion_pct", "endpoint_pct", "path_efficiency",
              "peak_speed_cm_s", "false_starts", "outcome", "onset_thr_cm", "target_r_cm", "home_r_cm"]


class LosTest:
    """HOME_WAIT (stand at centre) -> ARMED (random delay) -> TARGET (dot shown; clock starts) ->
    RETURN (dot gone; clock for the way back) -> next trial. All times come from the DAQ sample counter."""

    def __init__(self, a, rng=None):
        self.a = a
        self.rng = rng if rng is not None else np.random.default_rng(a.seed)
        self.lean = a.mode == "lean"
        n_dirs = getattr(a, "n_dirs", None) or (4 if self.lean else 8)
        d = a.dirs or ("kiri,kanan,atas,bawah" if n_dirs == 4 else "all")
        names = list(DIRS) if d == "all" else [x.strip() for x in d.split(",")]
        names = sorted(names, key=lambda n: CANON.index(n) if n in CANON else 99)
        self.cols = LEAN_COLS if self.lean else TRIAL_COLS
        bad = [d for d in names if d not in DIRS]
        if bad: raise SystemExit(f"unknown direction(s) {bad}; choose from {list(DIRS)}")
        self.names = names
        self.rows, self.queue, self.cur, self.total = [], [], None, 0
        self.state, self.false, self.hold0, self.home0, self.onset_at = "IDLE", 0, None, None, 0.0
        self.last_xy = (0.0, 0.0)
        self.t_rec0 = 0.0          # sample-clock time at which the raw recording starts (set by the UI at Start)
        self.subject = getattr(a, "subject", "") or ""
        self.paths = []            # (direction, array[t, x, y] at ~100 Hz) per finished trial, for the PDF report

    @property
    def active(self): return self.state not in ("IDLE", "DONE")
    @property
    def target_visible(self): return self.state == "TARGET"

    def target_xy(self, name):
        th = np.radians(DIRS[name]); k = self.a.target_pct / 100.0
        return self.a.los_x * k * np.cos(th), self.a.los_y * k * np.sin(th)

    def begin(self):
        q = self.names * self.a.reps
        if self.a.order == "random": self.rng.shuffle(q)
        self.queue, self.total, self.rows, self.cur = list(q), len(q), [], None
        self.paths = []
        self.false, self.hold0, self.state = 0, None, "HOME_WAIT"

    def stop(self): self.state = "IDLE"

    def step(self, t, sx, sy):
        """One valid CoP sample in screen cm relative to the home position, at sample time t (s)."""
        a = self.a; self.last_xy = (sx, sy)
        st, dh = self.state, np.hypot(sx, sy)
        if st == "HOME_WAIT":
            if dh <= a.home_r:
                if self.hold0 is None: self.hold0 = t
                elif t - self.hold0 >= a.hold_s:
                    self.onset_at = t + self.rng.uniform(a.delay_min, a.delay_max); self.state = "ARMED"
            else: self.hold0 = None
        elif st == "ARMED":
            if dh > a.home_r:                       # left home before the target appeared
                self.false += 1; self.state, self.hold0 = "HOME_WAIT", None
            elif t >= self.onset_at: self.state = "SHOW"      # target is drawn at the end of this frame
        elif st == "TARGET":
            c = self.cur; c["path"].append((t, sx, sy))
            if self.lean:                       # lean as far as you can: fixed window, then come back
                if t - c["t0"] >= a.lean_s: self.state, self.home0 = "RETURN", None
            else:
                if np.hypot(sx - c["tx"], sy - c["ty"]) <= a.target_r:
                    if c["in_t"] is None: c["in_t"] = t
                    elif t - c["in_t"] >= a.dwell_s: c["t_reach"] = c["in_t"]; self.state, self.home0 = "RETURN", None
                else: c["in_t"] = None
                if self.state == "TARGET" and t - c["t0"] > a.reach_timeout:
                    c["outcome"] = "target_timeout"; self.state, self.home0 = "RETURN", None
        elif st == "RETURN":
            c = self.cur; c["path"].append((t, sx, sy))
            if dh <= a.home_r:
                if self.home0 is None: self.home0 = t
                elif t - self.home0 >= a.dwell_s: c["t_home"] = self.home0; self._finish(); return
            else: self.home0 = None
            base = (c["t0"] + a.lean_s) if self.lean else (
                c["t_reach"] if c["t_reach"] is not None else c["t0"] + a.reach_timeout)
            if t - base > a.return_timeout:
                if c["outcome"] == "ok": c["outcome"] = "return_timeout"
                self._finish()

    def after_frame(self, t_end):
        """Call once per display frame, right after the samples were processed: stamps the target onset."""
        if self.state == "SHOW":
            name = self.queue.pop(0); tx, ty = self.target_xy(name)
            self.cur = dict(name=name, tx=tx, ty=ty, t0=t_end, p0=self.last_xy, path=[], in_t=None,
                            t_reach=None, t_home=None, outcome="ok", false=self.false)
            self.false, self.state = 0, "TARGET"

    def _finish(self):
        c = self.cur
        if c["path"]: self.paths.append((c["name"], np.array(c["path"])[::10]))
        self.rows.append(self._lean_row(c) if self.lean else self._target_row(c))
        self.cur = None
        self.state = "DONE" if not self.queue else "HOME_WAIT"
        self.hold0 = None

    def _target_row(self, c):
        a = self.a
        P = np.array(c["path"]) if c["path"] else np.empty((0, 3))
        t0, tr, th, nan = c["t0"], c["t_reach"], c["t_home"], np.nan
        tn, mxe, epe, eff, pk = nan, nan, nan, nan, nan
        Tn = np.hypot(c["tx"], c["ty"]); ux, uy = c["tx"] / Tn, c["ty"] / Tn
        if len(P):
            d = np.hypot(P[:, 1] - c["p0"][0], P[:, 2] - c["p0"][1])
            hit = np.nonzero(d >= a.onset_cm)[0]
            i_on = int(hit[0]) if len(hit) else None
            if i_on is not None: tn = P[i_on, 0]                          # movement onset
            m = (P[:, 0] <= tr) if tr is not None else np.ones(len(P), bool)   # up to first arrival
            proj = P[:, 1] * ux + P[:, 2] * uy
            mxe = proj.max() / Tn * 100                                       # whole trial, incl. any overshoot
            if m.any():
                epe = proj[m][-1] / Tn * 100
                if i_on is not None:
                    seg = P[i_on:np.nonzero(m)[0][-1] + 1]
                    if len(seg) > 2:
                        seg = seg[np.unique(np.searchsorted(seg[:, 0], np.arange(seg[0, 0], seg[-1, 0], 0.02)))]
                        pl = np.hypot(np.diff(seg[:, 1]), np.diff(seg[:, 2])).sum()
                        if pl > 0: eff = np.hypot(seg[-1, 1] - seg[0, 1], seg[-1, 2] - seg[0, 2]) / pl
            j = np.searchsorted(P[:, 0], P[:, 0] + 0.1); ok = j < len(P)  # peak speed over 100 ms windows
            if ok.any():
                pk = (np.hypot(P[j[ok], 1] - P[ok, 1], P[j[ok], 2] - P[ok, 2]) / (P[j[ok], 0] - P[ok, 0])).max()
        return dict(
            trial=len(self.rows) + 1, subject=self.subject, direction=c["name"], target_x_cm=c["tx"], target_y_cm=c["ty"],
            t_target_on_s=t0, t_cue_rec_s=t0 - self.t_rec0,
            reaction_s=tn - t0, time_to_target_s=(tr - t0) if tr is not None else nan,
            movement_s=(tr - tn) if tr is not None else nan,
            return_s=(th - tr) if (th is not None and tr is not None) else nan,
            cycle_s=(th - t0) if th is not None else nan, max_excursion_pct=mxe, endpoint_pct=epe,
            path_efficiency=eff, peak_speed_cm_s=pk, false_starts=c["false"], outcome=c["outcome"],
            onset_thr_cm=a.onset_cm, target_r_cm=a.target_r, home_r_cm=a.home_r)

    def _lean_row(self, c):
        """Metrics of one 'lean toward the light as far as you can' trial (window = lean_s after the cue)."""
        a, nan = self.a, np.nan
        t0 = c["t0"]
        P = np.array([p for p in c["path"] if p[0] <= t0 + a.lean_s]) if c["path"] else np.empty((0, 3))
        tn_ = np.hypot(c["tx"], c["ty"]); ux, uy = c["tx"] / tn_, c["ty"] / tn_   # direction of the lit light (= nominal angle if the ellipse is a circle)
        row = dict(trial=len(self.rows) + 1, subject=self.subject, direction=c["name"], direction_en=EN[c["name"]],
                   t_cue_s=t0, t_cue_rec_s=t0 - self.t_rec0,
                   reaction_ms=nan, velocity_m_s=nan, peak_speed_m_s=nan, max_excursion_m=nan, max_radial_m=nan,
                   direction_control_pct=nan, angle_deg=nan, movement_s=nan, false_starts=c["false"],
                   outcome=c["outcome"], onset_thr_cm=a.onset_cm, lean_s=a.lean_s)
        if len(P) < 3: row["outcome"] = "no_data"; return row
        rel = P[:, 1:3] - np.array(c["p0"])
        par = rel[:, 0] * ux + rel[:, 1] * uy             # cm along the intended direction
        per = -rel[:, 0] * uy + rel[:, 1] * ux            # cm sideways (off-axis)
        rad = np.hypot(rel[:, 0], rel[:, 1])
        row["max_excursion_m"], row["max_radial_m"] = par.max() / 100, rad.max() / 100
        if a.height_m: row["angle_deg"] = np.degrees(np.arctan2(par.max() / 100, 0.55 * a.height_m))
        hit = np.nonzero(rad >= a.onset_cm)[0]
        pk = par.max()                                     # "peak" = first time 95 % of the maximum is reached,
        i_pk = int(np.argmax(par >= 0.95 * pk)) if pk > 0 else 0     # so holding at the end does not move it
        if not len(hit) or i_pk <= int(hit[0]): row["outcome"] = "no_movement"; return row
        i_on = int(hit[0])
        row["reaction_ms"] = (P[i_on, 0] - t0) * 1000
        row["movement_s"] = P[i_pk, 0] - P[i_on, 0]
        row["velocity_m_s"] = (par[i_pk] - par[i_on]) / 100 / row["movement_s"]
        st, sp, sq = P[i_on:i_pk + 1, 0], par[i_on:i_pk + 1], per[i_on:i_pk + 1]
        idx = np.unique(np.searchsorted(st, np.arange(st[0], st[-1], 0.02)))     # 50 Hz: ignore sample jitter
        if len(idx) > 2:
            on_, off_ = np.abs(np.diff(sp[idx])).sum(), np.abs(np.diff(sq[idx])).sum()
            if on_ > 0: row["direction_control_pct"] = (on_ - off_) / on_ * 100
        X, Y = P[i_on:i_pk + 1, 1], P[i_on:i_pk + 1, 2]
        j = np.searchsorted(st, st + 0.1); ok = j < len(st)
        if ok.any(): row["peak_speed_m_s"] = (np.hypot(X[j[ok]] - X[ok], Y[j[ok]] - Y[ok]) / (st[j[ok]] - st[ok])).max() / 100
        return row

    def summary_table(self):
        """Report in the layout of the other software: direction blocks x metrics, one column per trial + mean."""
        mets = list(LEAN_METRICS) + ([("Maximal COP angle (deg)", "angle_deg", "{:.2f}")] if self.a.height_m else [])
        by = {d: [r for r in self.rows if r["direction"] == d] for d in self.names}
        n = max((len(v) for v in by.values()), default=0)
        hdr = [""] + [f"T{k + 1}" for k in range(n)] + ["Mean"]
        rows = []
        for d in self.names:
            if not by[d]: continue
            rows.append([EN[d]] + [""] * (n + 1))
            for label, key, fmt in mets:
                v = [r[key] for r in by[d]]
                good = [x for x in v if x == x]
                rows.append([label] + [fmt.format(x) if x == x else "-" for x in v] + [""] * (n - len(v)) +
                            [fmt.format(np.mean(good)) if good else "-"])
        return hdr, rows

    def summary_text(self):
        hdr, rows = self.summary_table()
        w0 = max(len(r[0]) for r in rows) if rows else 10
        return "\n".join(["  ".join([h.ljust(w0) if k == 0 else h.rjust(8) for k, h in enumerate(hdr)])] +
                         ["  ".join([c.ljust(w0) if k == 0 else c.rjust(8) for k, c in enumerate(r)]) for r in rows])

    def save_summary_csv(self, path):
        hdr, rows = self.summary_table()
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

    def summary_lines(self, n=4):
        out = [f"LOS test: trial {len(self.rows)}/{self.total}"]
        for r in self.rows[-n:]:
            if self.lean:
                out.append(f"#{r['trial']} {r['direction_en']:<9s} RT {r['reaction_ms']:.0f} ms  "
                           f"exc {r['max_excursion_m'] * 100:.1f} cm  DCL {r['direction_control_pct']:.0f}%" +
                           (f"  [{r['outcome']}]" if r["outcome"] != "ok" else ""))
                continue
            out.append(f"#{r['trial']} {r['direction']:<11s} reach {r['time_to_target_s']:.2f}s  "
                       f"back {r['return_s']:.2f}s  RT {r['reaction_s']:.2f}s" +
                       (f"  [{r['outcome']}]" if r["outcome"] != "ok" else ""))
        return out

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.cols); w.writeheader()
            for r in self.rows: w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    def means_text(self):
        if self.lean: return self.summary_text()
        lines = ["direction     n  reach_s  return_s  reaction_s  max_exc_%"]
        for name in self.names:
            rs = [r for r in self.rows if r["direction"] == name and r["outcome"] == "ok"]
            if rs:
                m = lambda k: np.nanmean([r[k] for r in rs])
                lines.append(f"{name:<12s} {len(rs):2d}  {m('time_to_target_s'):6.2f}  {m('return_s'):8.2f}  "
                             f"{m('reaction_s'):10.2f}  {m('max_excursion_pct'):9.0f}")
        return "\n".join(lines)


# ---------------- LOS display ----------------
CSV_HEADER = ("t_s,L_c1_N,L_c2_N,L_c3_N,L_c4_N,R_c1_N,R_c2_N,R_c3_N,R_c4_N,"
              "Fz_N,L_Fz_N,R_Fz_N,copx_m,copy_m")


# ---------------- settings form (python los_live.py with no arguments) ----------------
FORM = [  # (key, label, default, tab, note)
    ("subject", "Nama responden", "", 0, "dipakai sebagai nama folder & awalan file"),
    ("height_cm", "Tinggi badan (cm)", "", 0, "opsional; hanya untuk sudut COP (asumsi CoG = 0,55 x tinggi, belum tervalidasi)"),
    ("reps", "Pengulangan tiap arah", "3", 0, ""),
    ("order", "Urutan arah", "acak", 0, "acak / tetap"),
    ("lean_s", "Waktu untuk condong (detik)", "6", 0, "mode Lean"),
    ("reach_timeout", "Batas waktu capai target (detik)", "8", 0, "mode Target"),
    ("return_timeout", "Batas waktu kembali ke tengah (detik)", "15", 0, "mode Lean & Target"),
    ("los_x", "Setengah lebar elips LOS, kiri-kanan (cm)", "10", 1, "placeholder, belum dari norma"),
    ("los_y", "Setengah tinggi elips LOS, depan-belakang (cm)", "8", 1, "placeholder, belum dari norma"),
    ("target_pct", "Jarak lampu/target (% dari elips)", "100", 1, ""),
    ("target_r", "Radius target / lampu (cm)", "2", 1, "juga syarat 'tercapai' di mode Target"),
    ("home_r", "Radius lingkaran tengah / home (cm)", "2", 1, "syarat 'sudah kembali'"),
    ("hold_s", "Diam di tengah sebelum trial (detik)", "1", 1, ""),
    ("delay_min", "Jeda acak sebelum lampu menyala, min (detik)", "1", 1, ""),
    ("delay_max", "Jeda acak sebelum lampu menyala, maks (detik)", "3", 1, ""),
    ("onset_cm", "Ambang mulai bergerak (cm)", "1", 1, "menentukan reaction time; bandingkan hanya dengan ambang sama"),
    ("dwell_s", "Lama di dalam target/home agar dihitung (detik)", "0.1", 1, ""),
    ("fs", "Sampling rate (Hz)", "1000", 2, "harus sama dengan setting DAQ"),
    ("cutoff", "Low-pass (Hz, 0 = mati)", "10", 2, ""),
    ("min_fz", "Gaya total minimum agar CoP dihitung (N)", "50", 2, ""),
    ("tare_s", "Durasi tare (detik)", "", 2, "kosong = 1,5 detik"),
    ("seed", "Seed acak", "", 2, "kosong = berbeda tiap kali"),
]
FORM_DEFAULTS = {k: d for k, _, d, _, _ in FORM}
FORM_DEFAULTS.update(mode="lean", n_dirs="4", source="ni", ni="Dev1/ai0:7", replay="", terminal="rse",
                     flip_x=True, flip_y=True, outdir="output")


def form_to_argv(v):
    """Pure function: form values (strings) -> (argv list, list of error messages)."""
    err, av = [], []
    def num(k, label, lo=None, hi=None, req=True):
        t = str(v.get(k, "")).strip().replace(",", ".")
        if t == "":
            if req: err.append(f"{label}: wajib diisi")
            return None
        try: x = float(t)
        except ValueError: err.append(f"{label}: '{t}' bukan angka"); return None
        if (lo is not None and x < lo) or (hi is not None and x > hi): err.append(f"{label}: di luar rentang"); return None
        return x
    if v["source"] == "ni":
        if not str(v["ni"]).strip(): err.append("Channel NI wajib diisi")
        av += ["--ni", str(v["ni"]).strip()]
    else:
        if not str(v["replay"]).strip(): err.append("File replay wajib dipilih")
        av += ["--replay", str(v["replay"]).strip()]
    if v["mode"] != "preview" and not str(v["subject"]).strip(): err.append("Nama responden wajib diisi untuk tes")
    vals = {k: num(k, lab, lo, hi, req) for k, lab, lo, hi, req in [
        ("fs", "Sampling rate", 1, 100000, True), ("cutoff", "Low-pass", 0, None, True), ("min_fz", "Gaya minimum", 0, None, True),
        ("reps", "Pengulangan", 1, 100, True), ("lean_s", "Waktu condong", 0.5, 120, True),
        ("reach_timeout", "Batas capai", 0.5, 120, True), ("return_timeout", "Batas kembali", 0.5, 300, True),
        ("los_x", "Elips kiri-kanan", 1, 50, True), ("los_y", "Elips depan-belakang", 1, 50, True),
        ("target_pct", "Jarak target", 10, 200, True), ("target_r", "Radius target", 0.2, 20, True),
        ("home_r", "Radius home", 0.2, 20, True), ("hold_s", "Diam di tengah", 0, 60, True),
        ("delay_min", "Jeda min", 0, 60, True), ("delay_max", "Jeda maks", 0, 60, True),
        ("onset_cm", "Ambang mulai", 0.1, 10, True), ("dwell_s", "Lama dwell", 0, 10, True),
        ("height_cm", "Tinggi badan", 50, 250, False), ("tare_s", "Durasi tare", 0.2, 30, False), ("seed", "Seed", 0, None, False)]}
    if vals["delay_min"] is not None and vals["delay_max"] is not None and vals["delay_min"] > vals["delay_max"]:
        err.append("Jeda min harus <= jeda maks")
    if err: return None, err
    g = lambda k: f"{vals[k]:g}"
    av += ["--fs", g("fs"), "--cutoff", g("cutoff"), "--min-fz", g("min_fz"), "--terminal", v["terminal"],
           "--los-x", g("los_x"), "--los-y", g("los_y"), "--target-pct", g("target_pct"), "--target-r", g("target_r"),
           "--home-r", g("home_r"), "--hold-s", g("hold_s"), "--delay-min", g("delay_min"), "--delay-max", g("delay_max"),
           "--onset-cm", g("onset_cm"), "--dwell-s", g("dwell_s"), "--lean-s", g("lean_s"),
           "--reach-timeout", g("reach_timeout"), "--return-timeout", g("return_timeout"),
           "--reps", str(int(vals["reps"])), "--order", "random" if str(v["order"]).lower().startswith(("a", "r")) else "fixed",
           "--flip-x" if v["flip_x"] else "--no-flip-x", "--flip-y" if v["flip_y"] else "--no-flip-y",
           "--n-dirs", str(v["n_dirs"]), "--outdir", str(v["outdir"]).strip() or "output",
           "--subject", str(v["subject"]).strip()]
    if v["mode"] != "preview": av += ["--los-test", "--mode", v["mode"]]
    if vals["height_cm"] is not None: av += ["--height-m", f"{vals['height_cm'] / 100:g}"]
    if vals["tare_s"] is not None: av += ["--tare-s", g("tare_s")]
    if vals["seed"] is not None: av += ["--seed", str(int(vals["seed"]))]
    return av, []


def settings_form():
    """Tk form. Returns the argv list for main(), or None if cancelled. Last values are remembered in los_settings.json."""
    import json, os
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        raise SystemExit("tkinter tidak tersedia. Jalankan dengan argumen, mis.: python los_live.py --ni Dev1/ai0:7 --los-test --mode lean")
    store = os.path.join(os.path.dirname(os.path.abspath(__file__)), "los_settings.json")
    vals = dict(FORM_DEFAULTS)
    try:
        with open(store, encoding="utf-8") as f: vals.update({k: x for k, x in json.load(f).items() if k in vals})
    except Exception: pass
    root = tk.Tk(); root.title("LOS - pengaturan"); result = {"argv": None}
    nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=8, pady=8)
    tabs = [ttk.Frame(nb, padding=10) for _ in range(3)]
    for t_, name in zip(tabs, ["Responden & tes", "Ukuran & waktu", "Alat & output"]): nb.add(t_, text=name)
    var = {}
    def mk(k, init):
        var[k] = (tk.BooleanVar if isinstance(init, bool) else tk.StringVar)(value=init); return var[k]
    for k in FORM_DEFAULTS: mk(k, vals[k])
    row = [0, 0, 0]
    def line(tab, label, widget_fn, note=""):
        r = row[tab]; ttk.Label(tabs[tab], text=label).grid(row=r, column=0, sticky="w", pady=2)
        w = widget_fn(tabs[tab]); w.grid(row=r, column=1, sticky="we", padx=6, pady=2)
        if note: ttk.Label(tabs[tab], text=note, foreground="gray40", wraplength=330).grid(row=r, column=2, sticky="w")
        row[tab] += 1
    # tab 0: who and what
    for k, lab, d, tab, note in FORM:
        if tab == 0 and k == "subject": line(0, lab, lambda p, k=k: ttk.Entry(p, textvariable=var[k], width=28), note)
    def radios(p, k, opts):
        f = ttk.Frame(p)
        for txt, val in opts: ttk.Radiobutton(f, text=txt, variable=var[k], value=val).pack(anchor="w")
        return f
    line(0, "Mode tes", lambda p: radios(p, "mode", [("Lean: lampu menyala, condong sejauh mungkin", "lean"),
         ("Target: capai titik lalu kembali", "target"), ("Hanya tampilan CoP (tanpa tes)", "preview")]))
    line(0, "Jumlah arah", lambda p: radios(p, "n_dirs", [("4D: kiri, kanan, depan, belakang", "4"),
         ("8D: 4D + 4 diagonal", "8")]))
    for k, lab, d, tab, note in FORM:
        if tab == 0 and k != "subject": line(0, lab, lambda p, k=k: ttk.Entry(p, textvariable=var[k], width=12), note)
    for k, lab, d, tab, note in FORM:
        if tab in (1, 2): line(tab, lab, lambda p, k=k: ttk.Entry(p, textvariable=var[k], width=12), note)
    # tab 2: source, terminal, flips, folder
    line(2, "Sumber data", lambda p: radios(p, "source", [("Force plate (NI-DAQmx)", "ni"), ("Replay file txt (uji tanpa alat)", "replay")]))
    line(2, "Channel NI", lambda p: ttk.Entry(p, textvariable=var["ni"], width=18), "contoh Dev1/ai0:7 (lihat NI MAX)")
    def file_row(p, k, pick):
        f = ttk.Frame(p); ttk.Entry(f, textvariable=var[k], width=28).pack(side="left")
        ttk.Button(f, text="Pilih...", command=lambda: var[k].set(pick() or var[k].get())).pack(side="left", padx=4); return f
    line(2, "File replay", lambda p: file_row(p, "replay", lambda: filedialog.askopenfilename(filetypes=[("txt", "*.txt"), ("all", "*.*")])))
    line(2, "Mode terminal NI", lambda p: ttk.Combobox(p, textvariable=var["terminal"], values=["rse", "nrse", "diff", "default"], width=10, state="readonly"),
         "rse = default yang sudah Anda uji")
    line(2, "Cermin kiri-kanan di layar", lambda p: ttk.Checkbutton(p, variable=var["flip_x"]))
    line(2, "Cermin depan-belakang di layar", lambda p: ttk.Checkbutton(p, variable=var["flip_y"]))
    line(2, "Folder output", lambda p: file_row(p, "outdir", lambda: filedialog.askdirectory()), "hasil masuk ke <folder>/<nama responden>/")
    def go():
        v = {k: x.get() for k, x in var.items()}
        argv, errs = form_to_argv(v)
        if errs: messagebox.showerror("Periksa isian", "\n".join(errs)); return
        try:
            with open(store, "w", encoding="utf-8") as f: json.dump({**v, "subject": ""}, f, indent=1)   # name is not remembered
        except Exception: pass
        result["argv"] = argv; root.destroy()
    bar = ttk.Frame(root); bar.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(bar, text="Mulai", command=go).pack(side="right", padx=4)
    ttk.Button(bar, text="Batal", command=root.destroy).pack(side="right")
    root.bind("<Return>", lambda e: go())
    root.mainloop()
    return result["argv"]


def make_parser():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay"); g.add_argument("--ni")
    ap.add_argument("--fs", type=float, default=1000.0, help="sampling rate (Hz) -- CONFIRM, not in the export")
    ap.add_argument("--cutoff", type=float, default=10.0, help="low-pass Hz on load-cell forces (0 = off)")
    ap.add_argument("--los-x", type=float, default=10.0, help="LOS half-width ML in cm (placeholder)")
    ap.add_argument("--los-y", type=float, default=8.0, help="LOS half-height AP in cm (placeholder)")
    ap.add_argument("--terminal", choices=["rse", "nrse", "diff", "default"], default="rse",
                    help="NI input wiring mode. Your stream showed L = L_true - R (differential-like pairing): use rse/nrse")
    ap.add_argument("--fix-pairs", action="store_true",
                    help="software fallback ONLY if the terminal mode cannot be changed: L_i += R_i (same index)")
    ap.add_argument("--min-fz", type=float, default=MIN_FZ, help="N; CoP hidden below this total vertical force")
    ap.add_argument("--tare-s", type=float, default=None,
                    help="seconds of EMPTY plates averaged at start (default 1.5 for --ni, 0 for --replay)")
    ap.add_argument("--flip-x", action=argparse.BooleanOptionalAction, default=True,
                    help="mirror left/right on screen (default on; --no-flip-x to disable)")
    ap.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True,
                    help="mirror front/back on screen (default on; --no-flip-y to disable)")
    ap.add_argument("--los-test", action="store_true",
                    help="target test: a dot appears, times are measured to reach it and to come back home")
    ap.add_argument("--mode", choices=["target", "lean"], default="target",
                    help="target = reach a dot and come back; lean = one of 4 lights, lean toward it as far as you can")
    ap.add_argument("--lean-s", type=float, default=6.0, help="lean mode: seconds allowed to lean after the light")
    ap.add_argument("--height-m", type=float, default=None,
                    help="lean mode: body height (m) to report a COP angle (assumes CoG height = 0.55 x height; unverified)")
    ap.add_argument("--n-dirs", type=int, choices=[4, 8], default=None,
                    help="4 = left/right/up/down, 8 = plus the four diagonals (default 4 for lean, 8 for target)")
    ap.add_argument("--subject", default="", help="respondent name: used as sub-folder and file prefix")
    ap.add_argument("--dirs", default=None, help="comma list of: " + ", ".join(DIRS) +
                    " (default: all for --mode target, kiri,kanan,atas,bawah for --mode lean)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions of each direction")
    ap.add_argument("--order", choices=["random", "fixed"], default="random")
    ap.add_argument("--target-pct", type=float, default=100.0, help="target distance as %% of the LOS ellipse")
    ap.add_argument("--target-r", type=float, default=2.0, help="target radius (cm): reached when CoP is inside")
    ap.add_argument("--home-r", type=float, default=2.0, help="home radius (cm) around the centre")
    ap.add_argument("--hold-s", type=float, default=1.0, help="s standing still at home before a trial is armed")
    ap.add_argument("--dwell-s", type=float, default=0.1, help="s inside target/home to count as arrived")
    ap.add_argument("--delay-min", type=float, default=1.0, help="random wait before the target appears (s)")
    ap.add_argument("--delay-max", type=float, default=3.0)
    ap.add_argument("--onset-cm", type=float, default=1.0, help="movement onset = CoP moved this far from start (cm)")
    ap.add_argument("--reach-timeout", type=float, default=8.0)
    ap.add_argument("--return-timeout", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--trail-s", type=float, default=3.0)
    ap.add_argument("--outdir", default="output",
                    help="folder for all result files (CSV, PNG); created if missing. "
                         "Relative paths are relative to this script's folder")
    ap.add_argument("--debug", action="store_true", help="print forces / CoP once per second")
    ap.add_argument("--save-frames", type=int, default=0, help="headless test: render N PNG frames and exit")
    return ap


def finalize_args(a):
    """Resolve the output folder (<outdir>/<respondent>), create it, return the sanitised respondent name."""
    import os, re
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.outdir)
    subj = re.sub(r"[^\w\-]+", "_", a.subject.strip()).strip("_")
    a.subject = subj
    if subj: a.outdir = os.path.join(a.outdir, subj)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"Output folder: {a.outdir}")
    return subj


def main(argv=None):
    import sys
    if argv is None and len(sys.argv) == 1:         # started without arguments -> settings form
        argv = settings_form()
        if argv is None: return
    ap = make_parser()
    a = ap.parse_args(argv)
    import os, re, json
    subj = finalize_args(a)
    tare_s = a.tare_s if a.tare_s is not None else (1.5 if a.ni else 0.0)
    lt = LosTest(a) if a.los_test else None

    import matplotlib
    if a.save_frames: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Circle
    from matplotlib.widgets import Button

    src = ReplaySource(a.replay, a.fs) if a.replay else NISource(a.ni, a.fs, terminal=a.terminal)
    src.start()

    sos = butter(4, a.cutoff, fs=a.fs, output="sos") if a.cutoff > 0 else None
    zi = None
    flip = {"x": a.flip_x, "y": a.flip_y}
    R0 = max(a.los_x, a.los_y) * 1.4          # minimum half-range of the view (cm)
    view = {"R": R0, "auto": True}
    trail = collections.deque(maxlen=int(a.trail_s * a.fs))
    # default centre = midpoint between the two plates until the user presses Centre
    centre = np.array([PLATE_CENTRE_X["R"] / 2, 0.0])
    recent = collections.deque(maxlen=int(3 * a.fs))  # last 3 s of valid CoP for centring
    ex = {}                                            # direction -> max normalised excursion (measurement only)
    offset = np.zeros(8)                               # tare, N
    tare = {"left": int(tare_s * a.fs), "acc": []}
    meas = {"on": False, "n0": 0, "t_wall": 0.0, "last_file": "", "base": ""}
    rec = []                                           # recorded blocks during a measurement
    cnt = {"n": 0, "dropped": 0}
    last = {"fz": np.nan, "valid": False}
    dbg = {"t": 0.0}
    warn = {"on": False}

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.subplots_adjust(bottom=0.22)
    ax.set_xlim(-R0, R0); ax.set_ylim(-R0, R0); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("ML (cm)"); ax.set_ylabel("AP (cm)")
    ax.add_patch(Ellipse((0, 0), 2 * a.los_x, 2 * a.los_y, fill=False, lw=2, ec="tab:red"))
    ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
    home_patch = Circle((0, 0), a.home_r, fill=False, ls="--", ec="0.4", lw=1.5, visible=lt is not None)
    target_patch = Circle((0, 0), a.target_r, fc="tab:orange", ec="k", alpha=.85, visible=False, zorder=4)
    ax.add_patch(home_patch); ax.add_patch(target_patch)
    cue_patches = {}
    if lt is not None and lt.lean:                     # four dim lights; the active one turns orange
        pos = {n_: lt.target_xy(n_) for n_ in lt.names}
        dmin = min([np.hypot(p1[0] - p2[0], p1[1] - p2[1]) for i1, p1 in enumerate(pos.values())
                    for p2 in list(pos.values())[i1 + 1:]] or [1e9])
        cue_r = min(a.target_r * 1.5, 0.45 * dmin)                 # lights must not overlap (8 directions)
        for nme, (cx_, cy_) in pos.items():
            cue_patches[nme] = Circle((cx_, cy_), cue_r, fc="0.85", ec="0.5", zorder=3)
            ax.add_patch(cue_patches[nme])
    line, = ax.plot([], [], "-", c="tab:blue", lw=1, alpha=.6)
    dot, = ax.plot([], [], "o", c="tab:blue", ms=12)
    txt = ax.text(.02, .98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)

    def save_measurement():
        if not rec: return
        arr = np.vstack(rec)
        path = f"{meas['base']}_raw.csv"
        np.savetxt(path, arr, delimiter=",", header=CSV_HEADER, comments="", fmt="%.5f")
        meas["last_file"] = path
        print(f"saved {len(arr)} samples -> {path}")

    def update(_):
        nonlocal zi
        new = src.read()
        if len(new) and a.fix_pairs:
            new = new.copy(); new[:, :4] += new[:, 4:]
        if len(new) and tare["left"] > 0:                    # collecting the tare (plates must be empty)
            take = new[:tare["left"]]
            tare["acc"].append(take); tare["left"] -= len(take)
            new = new[len(take):]
            if tare["left"] == 0:
                offset[:] = np.vstack(tare["acc"]).mean(0); tare["acc"] = []; zi = None
                print("tare done, offsets (N):", offset.round(2))
        if len(new) and tare["left"] == 0:
            new = new - offset
            if sos is not None:
                if zi is None:   # initialise filter state at the first sample (no start-up transient)
                    zi = sosfilt_zi(sos)[:, :, None] * new[0][None, None, :]   # (sections, 2, channels)
                new, zi = sosfilt(sos, new, axis=0, zi=zi)
            c = cop_from_cells(new, a.min_fz)
            n = len(new)
            if meas["on"]:
                t = (cnt["n"] + np.arange(n) - meas["n0"]) / a.fs
                rec.append(np.column_stack([t, new, c["Fz"], c["L_Fz"], c["R_Fz"], c["copx"], c["copy"]]))
            cnt["n"] += n
            cnt["dropped"] += int(c["dropped"].sum())
            last["fz"], last["valid"] = c["Fz"][-1], not np.isnan(c["copx"][-1])
            if min(c["L_Fz"].min(), c["R_Fz"].min()) < -30 and not warn["on"]:
                warn["on"] = True
                print("WARNING: a plate shows NEGATIVE force (< -30 N). A plate can only be pushed down, so this means "
                      "the acquisition is wrong (terminal mode / channel order / tare while loaded). Try --terminal rse, "
                      "re-tare with EMPTY plates, and check with --debug.")
            n_before = cnt["n"] - n               # cnt["n"] was already advanced by n
            for i, (x, y) in enumerate(zip(c["copx"], c["copy"])):
                if np.isnan(x): continue
                recent.append((x, y)); trail.append((x, y))
                if lt is not None and lt.active:  # same transform as the display: centre, then flips
                    sx = (x - centre[0]) * 100.0 * (-1 if flip["x"] else 1)
                    sy = (y - centre[1]) * 100.0 * (-1 if flip["y"] else 1)
                    lt.step((n_before + i) / a.fs, sx, sy)
            if not last["valid"]:
                trail.clear()                                 # stepped off / unloaded: drop stale trail
            if a.debug and time.time() - dbg["t"] > 1.0:
                dbg["t"] = time.time()
                print(f"cells N {np.round(new[-1], 1)}  Fz {c['Fz'][-1]:.1f}  L {c['L_Fz'][-1]:.1f}  "
                      f"R {c['R_Fz'][-1]:.1f}  copx {c['copx'][-1]*100:.1f} cm  copy {c['copy'][-1]*100:.1f} cm  "
                      f"dropped {cnt['dropped']}")

        if tare["left"] > 0:
            status = f"TARING... keep plates EMPTY ({tare['left']/a.fs:.1f} s)"
        elif meas["on"]:
            status = f"MEASURING {time.time() - meas['t_wall']:.1f} s  (space = stop & save)"
        else:
            status = "PREVIEW  (space = start measurement)"
        if warn["on"]: status += "\n!! NEGATIVE PLATE FORCE: check --terminal / tare (see terminal)"
        if lt is not None:
            lt.after_frame(cnt["n"] / a.fs)                      # stamps the target onset (sample clock)
            if lt.state == "DONE" and meas["on"]: finish_los()
            if lt.lean:
                for nme, p in cue_patches.items():
                    on = lt.target_visible and lt.cur is not None and lt.cur["name"] == nme
                    p.set_facecolor("tab:orange" if on else "0.85")
                target_patch.set_visible(False)
            else:
                target_patch.set_visible(lt.target_visible)
                if lt.cur is not None: target_patch.center = (lt.cur["tx"], lt.cur["ty"])
            ret = lt.state == "RETURN"
            home_patch.set_edgecolor("tab:green" if ret else "0.4"); home_patch.set_linewidth(4 if ret else 1.5)
            if meas["on"]:
                msg = {"HOME_WAIT": "stand still at the centre (dashed circle)", "ARMED": "wait... do not move",
                       "SHOW": "", "TARGET": (">>> LEAN TO THE LIGHT AS FAR AS YOU CAN <<<" if lt.lean else ">>> GO TO THE TARGET <<<"),
                       "RETURN": "<<< RETURN TO THE CENTRE >>>"}
                status += "\n" + msg.get(lt.state, "") + "\n" + "\n".join(lt.summary_lines())

        if not trail or not last["valid"]:
            dot.set_data([], []); line.set_data([], [])
            txt.set_text(f"{status}\nno valid CoP: Fz {last['fz']:.0f} N < {a.min_fz:.0f} N\n"
                         f"rejected samples: {cnt['dropped']}" + (f"\nlast file: {meas['last_file']}" if meas["last_file"] else ""))
            return line, dot, txt
        T = np.array(trail)
        xy = (T - centre) * 100.0
        if flip["x"]: xy[:, 0] *= -1
        if flip["y"]: xy[:, 1] *= -1
        if view["auto"]:   # grow instantly, shrink slowly -> follows the trail, a spike cannot stick
            need = max(R0, np.nanmax(np.abs(xy)) * 1.15)
            view["R"] = need if need > view["R"] else max(need, view["R"] * 0.98)
        if ax.get_xlim()[1] != view["R"]:
            ax.set_xlim(-view["R"], view["R"]); ax.set_ylim(-view["R"], view["R"])
        line.set_data(xy[:, 0], xy[:, 1]); dot.set_data([xy[-1, 0]], [xy[-1, 1]])
        nx, ny = xy[-1, 0] / a.los_x, xy[-1, 1] / a.los_y
        r = np.hypot(nx, ny)
        if meas["on"]:
            d = int(np.round(np.degrees(np.arctan2(ny, nx)) / 45)) % 8
            ex[d] = max(ex.get(d, 0), r)
        names = ["ML+", "AP+/ML+", "AP+", "AP+/ML-", "ML-", "AP-/ML-", "AP-", "AP-/ML+"]
        txt.set_text(f"{status}\nCoP  ML {xy[-1,0]:+5.1f} cm  AP {xy[-1,1]:+5.1f} cm   Fz {last['fz']:.0f} N\n"
                     f"{'LIMIT REACHED' if r >= 1 else f'{r*100:5.0f}% of limit'}\n" +
                     "\n".join(f"{names[k]:8s} max {v*100:4.0f}%" for k, v in sorted(ex.items())) +
                     f"\nrejected samples: {cnt['dropped']}" +
                     (f"\nlast file: {meas['last_file']}" if meas["last_file"] else ""))
        dot.set_color("tab:green" if r >= 1 else "tab:blue")
        return line, dot, txt

    # ---- actions (shared by on-screen buttons and keyboard) ----
    def do_tare(*_):
        nonlocal zi
        if meas["on"]: print("stop the measurement before taring"); return
        tare.update(left=int(max(tare_s, 1.0) * a.fs), acc=[]); zi = None; trail.clear(); recent.clear()
        print("taring: keep plates EMPTY")
    def save_trials():
        if lt is None or not lt.rows: return
        path = f"{meas['base']}_trials.csv"
        lt.save_csv(path); meas["last_file"] = path
        print(f"saved {len(lt.rows)} trials -> {path}\n{lt.means_text()}")
        if lt.lean:
            lt.save_summary_csv(f"{meas['base']}_summary.csv")
            try:                                       # report window + PNG, like the other software's table
                hdr, rows = lt.summary_table()
                f2, ax2 = plt.subplots(figsize=(max(6.0, 2.8 + 1.0 * len(hdr)), min(0.34 * len(rows) + 1.2, 10.0))); ax2.axis("off")
                tb = ax2.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="left",
                               colWidths=[0.42] + [0.58 / (len(hdr) - 1)] * (len(hdr) - 1))
                tb.auto_set_font_size(False); tb.set_fontsize(9 if len(rows) <= 30 else 7); tb.scale(1, 1.25 if len(rows) <= 30 else 1.0)
                if subj: f2.suptitle(f"{subj}   ({len(lt.names)} directions)", fontsize=10, y=0.995); f2.subplots_adjust(top=0.94)
                f2.savefig(f"{meas['base']}_summary.png", dpi=110)
                if not a.save_frames: f2.show()
            except Exception as e: print("summary window failed:", e)
    def finish_los():
        print("LOS test finished"); start_stop()
    def start_stop(*_):
        nonlocal centre
        if tare["left"] > 0: print("wait for tare to finish"); return
        if not meas["on"]:
            if lt is not None:
                if len(recent) < int(0.5 * a.fs): print("LOS test: step on the plates and stand still first"); return
                centre = np.mean(list(recent)[-int(a.fs):], axis=0)     # home = quiet-standing CoP of the last 1 s
                lt.begin(); print("home set (m):", centre.round(4))
            meas["base"] = os.path.join(a.outdir, (f"{subj}_" if subj else "") + f"{datetime.datetime.now():%Y%m%d_%H%M%S}")
            if lt is not None: lt.t_rec0 = cnt["n"] / a.fs
            try:                                   # every setting used, next to the data
                with open(f"{meas['base']}_params.json", "w", encoding="utf-8") as fp:
                    json.dump({k: v for k, v in vars(a).items()}, fp, indent=1, ensure_ascii=False, default=str)
            except Exception as e: print("params file failed:", e)
            rec.clear(); ex.clear(); meas.update(on=True, n0=cnt["n"], t_wall=time.time()); btn_ss.label.set_text("Stop (space)")
            print("measurement started")
        else:
            meas["on"] = False; btn_ss.label.set_text("Start (space)"); save_measurement()
            if lt is not None:
                done = lt.state == "DONE"; lt.stop(); save_trials()
                if done: lt.state = "DONE"
    def set_centre(*_):
        nonlocal centre
        if recent: centre = np.mean(recent, axis=0); print("centre set (m):", centre.round(4))
    def reset(*_): ex.clear(); view.update(R=R0, auto=True)
    def toggle_auto(*_): view["auto"] = not view["auto"]; print("autoscale", view["auto"])
    def zoom_in(*_): view.update(R=view["R"] / 1.25, auto=False)
    def zoom_out(*_): view.update(R=view["R"] * 1.25, auto=False)
    def flip_x(*_): flip["x"] = not flip["x"]; print("flip X (left/right):", flip["x"])
    def flip_y(*_): flip["y"] = not flip["y"]; print("flip Y (front/back):", flip["y"])

    keymap = {"t": do_tare, " ": start_stop, "c": set_centre, "r": reset, "a": toggle_auto,
              "+": zoom_in, "=": zoom_in, "plus": zoom_in, "-": zoom_out, "_": zoom_out, "minus": zoom_out,
              "x": flip_x, "y": flip_y, "q": lambda *_: plt.close(fig)}
    def on_key(e):
        f = keymap.get(e.key)
        if f: f()
    for k in ("keymap.back", "keymap.home", "keymap.forward", "keymap.yscale", "keymap.xscale",
              "keymap.grid", "keymap.fullscreen", "keymap.save"):   # free up c, r, a, x, y ...
        try: plt.rcParams[k] = []
        except Exception: pass
    fig.canvas.mpl_connect("key_press_event", on_key)

    btns = []
    def add_btn(x, y, w, label, fn):
        b = Button(fig.add_axes([x, y, w, 0.06]), label); b.on_clicked(fn); btns.append(b); return b
    add_btn(0.02, 0.10, 0.23, "Tare (t)", do_tare)
    btn_ss = add_btn(0.27, 0.10, 0.23, "Start (space)", start_stop)
    add_btn(0.52, 0.10, 0.23, "Centre (c)", set_centre)
    add_btn(0.77, 0.10, 0.21, "Reset (r)", reset)
    for n, (label, fn) in enumerate([("Auto (a)", toggle_auto), ("Zoom +", zoom_in), ("Zoom -", zoom_out),
                                     ("Flip L/R (x)", flip_x), ("Flip F/B (y)", flip_y)]):
        add_btn(0.02 + n * 0.196, 0.02, 0.185, label, fn)
    try: fig.canvas.manager.set_window_title("LOS live v7")
    except Exception: pass
    print(f"los_live v7: subject={subj or '-'} los_test={a.los_test} flip_x={flip['x']} flip_y={flip['y']} min_fz={a.min_fz:.0f} N tare={tare_s:.1f} s")

    if a.save_frames:
        for k in range(a.save_frames):
            time.sleep(0.5); update(None)
            if k == 1: set_centre()
        do_tare(); time.sleep(1.2); update(None); update(None)       # mechanics only (replay is loaded)
        start_stop(); time.sleep(0.6); update(None); update(None); start_stop()
        for f in (flip_x, flip_y, zoom_in, zoom_out, toggle_auto, toggle_auto, flip_x, flip_y, reset):
            f(); update(None)
        on_key(type("E", (), {"key": "x"})()); on_key(type("E", (), {"key": "+"})()); update(None)
        print("after test keys: flip", flip, "view", view)
        if lt is not None:                                   # show a target in the picture
            lt.begin(); lt.state = "SHOW"; meas["on"] = True; update(None); meas["on"] = False
        fig.savefig("los_test.png", dpi=90)
        return
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    try: plt.show()
    finally:
        if meas["on"]: save_measurement()
        src.stop()


if __name__ == "__main__":
    main()
