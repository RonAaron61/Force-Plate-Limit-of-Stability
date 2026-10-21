#!/usr/bin/env python3
"""
LOS app: one window for the whole Limits-of-Stability session.
  1) Setup page   : subject, test mode (lean / target / CoP only), 4D or 8D, repetitions, save folder, ...
  2) Measure page : live CoP, tare / start / stop, results table, PDF report.
Needs los_live.py and los_report.py in the same folder.   Run:  python los_app.py
"""
import collections, datetime, json, os, subprocess, sys, time
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Circle
import los_live as L
import los_report

VERSION = "Contemplas LOS app v8"
HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "los_app_settings.json")
BG, CARD, INK, MUTED, ACCENT, ACCENT_H = "#eef1f5", "#ffffff", "#1b2733", "#6b7785", "#1f4e79", "#2a6aa6"
OK, WARN, BAD, LINE, BLUE = "#2e9e5b", "#e08a1e", "#c0392b", "#d5dbe3", "#1f6fb2"

# ---- dropdown options (label shown -> value used) ----
MODES = {"Lean: condong ke lampu": "lean", "Target: capai titik, kembali": "target", "Hanya CoP (tanpa tes)": "preview"}
NDIRS = {"4D (kiri, kanan, depan, belakang)": "4", "8D (4D + 4 diagonal)": "8"}
ORDERS = {"Acak": "acak", "Tetap": "tetap"}
UPPER = {"Di pinggang": "On hips", "Bebas di samping": "Free at sides", "Menyilang di dada": "Crossed on chest"}
STANCE = {"Paralel": "Parallel", "Semi-tandem": "Semi-tandem", "Tandem": "Tandem"}
FOOT = {"Tanpa alas kaki": "Barefoot", "Bersepatu": "Shoes"}
SOURCES = {"Force plate (NI-DAQmx)": "ni", "Replay file txt (uji tanpa alat)": "replay"}
REPS = ["1", "2", "3", "4", "5", "6", "8", "10"]
LEAN_S = ["3", "4", "5", "6", "8", "10"]
ADV = [  # key, label, presets, note   (editable dropdowns: pick a preset or type a value)
    ("los_x", "Setengah lebar elips LOS, kiri-kanan (cm)", ["8", "10", "12", "15"], "placeholder, belum dari norma"),
    ("los_y", "Setengah tinggi elips LOS, depan-belakang (cm)", ["6", "8", "10", "12"], "placeholder, belum dari norma"),
    ("target_pct", "Jarak lampu/target (% dari elips)", ["100", "75", "50"], ""),
    ("target_r", "Radius lampu/target (cm)", ["1.5", "2", "3"], "juga syarat 'tercapai' di mode Target"),
    ("home_r", "Radius lingkaran tengah (cm)", ["1.5", "2", "2.5", "3"], "syarat 'sudah kembali'"),
    ("hold_s", "Diam di tengah sebelum trial (detik)", ["0.5", "1", "2"], ""),
    ("delay_min", "Jeda acak sebelum lampu, min (detik)", ["0.5", "1", "2"], ""),
    ("delay_max", "Jeda acak sebelum lampu, maks (detik)", ["2", "3", "5"], ""),
    ("onset_cm", "Ambang mulai bergerak (cm)", ["0.5", "1", "1.5", "2"], "menentukan reaction time"),
    ("dwell_s", "Lama di dalam target/home (detik)", ["0.05", "0.1", "0.2"], ""),
    ("reach_timeout", "Batas waktu capai target (detik)", ["5", "8", "10", "15"], "mode Target"),
    ("return_timeout", "Batas waktu kembali (detik)", ["10", "15", "20", "30"], ""),
    ("fs", "Sampling rate (Hz)", ["500", "1000", "2000"], "harus sama dengan setting DAQ"),
    ("cutoff", "Low-pass (Hz, 0 = mati)", ["0", "5", "10", "15", "20"], "filter 10 Hz menunda sinyal sekitar 40 ms"),
    ("min_fz", "Gaya total minimum agar CoP dihitung (N)", ["30", "50", "100"], ""),
    ("tare_s", "Durasi tare (detik, kosong = 1,5)", ["", "1.5", "2", "3"], ""),
    ("seed", "Seed acak (kosong = berbeda tiap kali)", ["", "1", "2", "3"], ""),
]


def open_file(path):
    try:
        if sys.platform.startswith("win"): os.startfile(path)
        elif sys.platform == "darwin": subprocess.Popen(["open", path])
        else: subprocess.Popen(["xdg-open", path])
    except Exception as e: messagebox.showerror("Gagal membuka", f"{path}\n{e}")


def make_source(a):
    return L.ReplaySource(a.replay, a.fs) if a.replay else L.NISource(a.ni, a.fs, terminal=a.terminal)


# ============================================================ signal engine (same maths as los_live.main)
class Engine:
    def __init__(self, a, src):
        self.a, self.src = a, src
        self.sos = butter(4, a.cutoff, fs=a.fs, output="sos") if a.cutoff > 0 else None
        self.zi = None
        self.tare_s = a.tare_s if a.tare_s is not None else (1.5 if a.ni else 0.0)
        self.tare = {"left": int(self.tare_s * a.fs), "acc": []}
        self.offset = np.zeros(8)
        self.centre = np.array([L.PLATE_CENTRE_X["R"] / 2, 0.0])
        self.flip = {"x": a.flip_x, "y": a.flip_y}
        self.trail = collections.deque(maxlen=int(a.trail_s * a.fs))
        self.recent = collections.deque(maxlen=int(3 * a.fs))
        self.sig = collections.deque(maxlen=300)           # (t, Fz, L, R) once per poll, for the Signals tab
        self.cnt, self.dropped = 0, 0
        self.fz, self.lfz, self.rfz, self.valid, self.warn = np.nan, np.nan, np.nan, False, False
        self.meas_on, self.n0, self.rec = False, 0, []
        self.lt = L.LosTest(a) if a.los_test else None

    @property
    def t_now(self): return self.cnt / self.a.fs

    def retare(self):
        self.tare.update(left=int(max(self.tare_s, 1.0) * self.a.fs), acc=[]); self.zi = None; self.warn = False
        self.trail.clear(); self.recent.clear(); self.valid = False

    def poll(self):
        a = self.a
        new = self.src.read()
        if len(new) and a.fix_pairs:
            new = new.copy(); new[:, :4] += new[:, 4:]
        if len(new) and self.tare["left"] > 0:
            take = new[:self.tare["left"]]
            self.tare["acc"].append(take); self.tare["left"] -= len(take); new = new[len(take):]
            if self.tare["left"] == 0:
                self.offset = np.vstack(self.tare["acc"]).mean(0); self.tare["acc"] = []; self.zi = None
        if len(new) and self.tare["left"] == 0:
            new = new - self.offset
            if self.sos is not None:
                if self.zi is None: self.zi = sosfilt_zi(self.sos)[:, :, None] * new[0][None, None, :]
                new, self.zi = sosfilt(self.sos, new, axis=0, zi=self.zi)
            c = L.cop_from_cells(new, a.min_fz)
            n = len(new); n_before = self.cnt
            if self.meas_on:
                t = (n_before + np.arange(n) - self.n0) / a.fs
                self.rec.append(np.column_stack([t, new, c["Fz"], c["L_Fz"], c["R_Fz"], c["copx"], c["copy"]]))
            self.cnt += n; self.dropped += int(c["dropped"].sum())
            self.fz, self.lfz, self.rfz = c["Fz"][-1], c["L_Fz"][-1], c["R_Fz"][-1]
            self.valid = not np.isnan(c["copx"][-1])
            self.sig.append((self.t_now, self.fz, self.lfz, self.rfz))
            if min(c["L_Fz"].min(), c["R_Fz"].min()) < -30: self.warn = True
            lt = self.lt
            for i, (x, y) in enumerate(zip(c["copx"], c["copy"])):
                if np.isnan(x): continue
                self.recent.append((x, y)); self.trail.append((x, y))
                if lt is not None and lt.active:
                    sx = (x - self.centre[0]) * 100.0 * (-1 if self.flip["x"] else 1)
                    sy = (y - self.centre[1]) * 100.0 * (-1 if self.flip["y"] else 1)
                    lt.step((n_before + i) / a.fs, sx, sy)
            if not self.valid: self.trail.clear()
        if self.lt is not None: self.lt.after_frame(self.cnt / a.fs)

    def screen_xy(self):
        T = np.array(self.trail)
        xy = (T - self.centre) * 100.0
        if self.flip["x"]: xy[:, 0] *= -1
        if self.flip["y"]: xy[:, 1] *= -1
        return xy


# ============================================================ the application
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(VERSION); self.configure(bg=BG); self.geometry("1280x800"); self.minsize(1100, 700)
        try:
            import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass
        fam = "Segoe UI" if "Segoe UI" in tkfont.families() else ("Helvetica" if sys.platform == "darwin" else "DejaVu Sans")
        self.fam = fam
        for n in ("TkDefaultFont", "TkTextFont", "TkMenuFont"): tkfont.nametofont(n).configure(family=fam, size=10)
        self._style()
        self.vars = {}
        self.page = None
        self.show_setup()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _style(self):
        st = ttk.Style(self); st.theme_use("clam"); f = self.fam
        st.configure(".", background=BG, foreground=INK, font=(f, 10))
        st.configure("Card.TFrame", background=BG)
        st.configure("TLabel", background=BG); st.configure("Card.TLabel", background=BG)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED, font=(f, 9))
        st.configure("H1.TLabel", font=(f, 22, "bold"), foreground=ACCENT)
        st.configure("Sub.TLabel", foreground=MUTED)
        st.configure("TLabelframe", background=BG, bordercolor=LINE, relief="solid", borderwidth=1)
        st.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=(f, 10, "bold"))
        st.configure("TButton", padding=(12, 7), background="#e3e8ef", bordercolor=LINE, font=(f, 10))
        st.map("TButton", background=[("active", "#d3dbe6"), ("disabled", "#f1f3f6")], foreground=[("disabled", "#9aa5b1")])
        st.configure("Accent.TButton", background=ACCENT, foreground="white", font=(f, 11, "bold"), padding=(18, 9), bordercolor=ACCENT)
        st.map("Accent.TButton", background=[("active", ACCENT_H), ("disabled", "#9fb3c8")], foreground=[("disabled", "white")])
        st.configure("Stop.TButton", background=BAD, foreground="white", font=(f, 11, "bold"), padding=(18, 9), bordercolor=BAD)
        st.map("Stop.TButton", background=[("active", "#d64a3a"), ("disabled", "#e3b3ad")], foreground=[("disabled", "white")])
        st.configure("TCombobox", padding=4, fieldbackground="white")
        st.configure("TEntry", padding=4, fieldbackground="white")
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(18, 8), background="#dde3ea", font=(f, 10, "bold"))
        st.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT)])
        st.configure("TCheckbutton", background=BG); st.configure("TRadiobutton", background=BG)
        st.configure("Small.TButton", padding=(4, 5))
        st.configure("Treeview", rowheight=26, font=(f, 10), fieldbackground="white"); st.configure("Treeview.Heading", font=(f, 10, "bold"), background="#dde3ea", padding=6)

    # ---------------------------------------------------------------- setup page
    def show_setup(self):
        if self.page is not None: self.page.destroy()
        self.page = ttk.Frame(self, padding=24); self.page.pack(fill="both", expand=True)
        saved = {}
        try:
            with open(SETTINGS, encoding="utf-8") as f: saved = json.load(f)
        except Exception: pass
        d = dict(L.FORM_DEFAULTS); d.update(order="Acak", mode_l=list(MODES)[0], ndirs_l=list(NDIRS)[0], reps="3", lean_s="6",
                                          upper_l=list(UPPER)[0], stance_l=list(STANCE)[0], foot_l=list(FOOT)[0],
                                          source_l=list(SOURCES)[0], outdir=os.path.join(HERE, "output"))
        d.update({k: v for k, v in saved.items() if k in d or k in ("flip_x", "flip_y")})
        self.vars = {k: (tk.BooleanVar(value=v) if isinstance(v, bool) else tk.StringVar(value=str(v))) for k, v in d.items()}
        V = self.vars
        ttk.Label(self.page, text="Limits of Stability", style="H1.TLabel").pack(anchor="w")
        ttk.Label(self.page, text="Pengaturan pengukuran. Pastikan plate KOSONG sebelum menekan Mulai (tare otomatis dilakukan di awal).",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 14))
        cols = ttk.Frame(self.page); cols.pack(fill="x")
        def card(title, c):
            fr = ttk.LabelFrame(cols, text=title, padding=14); fr.grid(row=0, column=c, sticky="nsew", padx=(0 if c == 0 else 12, 0))
            cols.columnconfigure(c, weight=1, uniform="c"); fr.columnconfigure(1, weight=1); return fr
        def field(fr, r, label, widget):
            ttk.Label(fr, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=6, padx=(0, 10))
            widget.grid(row=r, column=1, sticky="we", pady=6)
        def dd(fr, key, values, editable=False):
            return ttk.Combobox(fr, textvariable=V[key], values=list(values), state="normal" if editable else "readonly", width=34)
        c1 = card("Subjek", 0)
        field(c1, 0, "Nama subjek", ttk.Entry(c1, textvariable=V["subject"], width=36))
        field(c1, 1, "Tinggi badan (cm)", dd(c1, "height_cm", ["", "150", "155", "160", "165", "170", "175", "180"], True))
        field(c1, 2, "Posisi tangan", dd(c1, "upper_l", UPPER))
        field(c1, 3, "Posisi kaki", dd(c1, "stance_l", STANCE))
        field(c1, 4, "Alas kaki", dd(c1, "foot_l", FOOT))
        ttk.Label(c1, text="Tinggi badan hanya dipakai untuk sudut COP (opsional).\nPosisi tangan/kaki/alas kaki hanya dicatat di laporan.",
                  style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        c2 = card("Tes", 1)
        field(c2, 0, "Mode", dd(c2, "mode_l", MODES))
        field(c2, 1, "Jumlah arah", dd(c2, "ndirs_l", NDIRS))
        field(c2, 2, "Pengulangan per arah", dd(c2, "reps", REPS, True))
        field(c2, 3, "Urutan arah", dd(c2, "order", ORDERS))
        field(c2, 4, "Waktu condong (detik)", dd(c2, "lean_s", LEAN_S, True))
        ttk.Button(c2, text="Pengaturan lanjutan...", command=self.advanced).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))
        c3 = card("Alat & penyimpanan", 2)
        field(c3, 0, "Sumber data", dd(c3, "source_l", SOURCES))
        field(c3, 1, "Channel NI", ttk.Entry(c3, textvariable=V["ni"]))
        f = ttk.Frame(c3, style="Card.TFrame"); ttk.Entry(f, textvariable=V["replay"]).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="...", width=3, command=lambda: V["replay"].set(filedialog.askopenfilename(filetypes=[("txt", "*.txt"), ("all", "*.*")]) or V["replay"].get())).pack(side="left", padx=(4, 0))
        field(c3, 2, "File replay", f)
        field(c3, 3, "Mode terminal NI", dd(c3, "terminal", ["rse", "nrse", "diff", "default"]))
        f = ttk.Frame(c3, style="Card.TFrame"); ttk.Entry(f, textvariable=V["outdir"]).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="...", width=3, command=lambda: V["outdir"].set(filedialog.askdirectory() or V["outdir"].get())).pack(side="left", padx=(4, 0))
        field(c3, 4, "Folder penyimpanan", f)
        ttk.Label(c3, text="Hasil disimpan di <folder>/<nama subjek>/", style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        bar = ttk.Frame(self.page); bar.pack(fill="x", pady=22)
        ttk.Button(bar, text="Keluar", command=self.on_close).pack(side="right")
        ttk.Button(bar, text="Mulai", style="Accent.TButton", command=self.go).pack(side="right", padx=10)
        self.in_setup = True
        self.bind("<Return>", lambda e: self.go() if self.in_setup else None)

    def advanced(self):
        top = tk.Toplevel(self); top.title("Pengaturan lanjutan"); top.configure(bg=BG); top.transient(self); top.grab_set()
        fr = ttk.Frame(top, padding=16); fr.pack(fill="both", expand=True)
        for r, (k, lab, presets, note) in enumerate(ADV):
            ttk.Label(fr, text=lab).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
            ttk.Combobox(fr, textvariable=self.vars[k], values=presets, width=10).grid(row=r, column=1, pady=3)
            ttk.Label(fr, text=note, style="Sub.TLabel").grid(row=r, column=2, sticky="w", padx=10)
        r = len(ADV)
        ttk.Checkbutton(fr, text="Cermin kiri-kanan di layar", variable=self.vars["flip_x"]).grid(row=r, column=0, sticky="w", pady=(10, 0))
        ttk.Checkbutton(fr, text="Cermin depan-belakang di layar", variable=self.vars["flip_y"]).grid(row=r + 1, column=0, sticky="w")
        ttk.Label(fr, text="Nilai bisa dipilih dari daftar atau diketik.", style="Sub.TLabel").grid(row=r + 2, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(fr, text="Selesai", style="Accent.TButton", command=top.destroy).grid(row=r + 3, column=2, sticky="e", pady=(12, 0))

    def form_values(self):
        V = {k: (x.get() if isinstance(x, tk.BooleanVar) else x.get().strip()) for k, x in self.vars.items()}
        V["mode"] = MODES[V["mode_l"]]; V["n_dirs"] = NDIRS[V["ndirs_l"]]; V["source"] = SOURCES[V["source_l"]]
        return V

    def go(self):
        V = self.form_values()
        argv, errs = L.form_to_argv(V)
        if errs: messagebox.showerror("Periksa isian", "\n".join(errs)); return
        keep = {k: x.get() for k, x in self.vars.items() if k != "subject"}
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f: json.dump(keep, f, indent=1)
        except Exception: pass
        a = L.make_parser().parse_args(argv)
        name = V["subject"]
        a.outdir = V["outdir"]
        L.finalize_args(a)
        meta = dict(name=name, height_cm=float(V["height_cm"].replace(",", ".")) if V["height_cm"] else None,
                    upper=UPPER[V["upper_l"]], stance=STANCE[V["stance_l"]], footwear=FOOT[V["foot_l"]], version=VERSION)
        try:
            self.page.destroy(); self.in_setup = False
            self.page = MeasurePage(self, a, meta); self.page.pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Tidak bisa memulai", f"{type(e).__name__}: {e}")
            self.show_setup()

    def back_to_setup(self):
        if isinstance(self.page, MeasurePage): self.page.shutdown()
        self.show_setup()

    def on_close(self):
        if isinstance(self.page, MeasurePage): self.page.shutdown()
        self.destroy()


# ============================================================ measurement page
class MeasurePage(ttk.Frame):
    def __init__(self, app, a, meta):
        super().__init__(app, padding=10)
        self.app, self.a, self.meta = app, a, meta
        self.src = make_source(a)
        self.src.start()                       # raises here if the DAQ cannot be opened -> caught by App.go
        self.eng = Engine(a, self.src)
        self.base, self.last_pdf, self.finished, self.alive = "", "", False, True
        self.view = {"R": max(a.los_x, a.los_y) * 1.55, "auto": True}
        self.R0 = self.view["R"]
        self.build()
        self.after(33, self.tick)

    # ---- layout
    def build(self):
        a, eng, lt = self.a, self.eng, self.eng.lt
        self.columnconfigure(1, weight=1); self.rowconfigure(1, weight=1)
        top = ttk.Frame(self); top.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 8))
        ttk.Label(top, text=self.meta["name"] or "(tanpa nama)", font=(self.app.fam, 16, "bold"), foreground=ACCENT).pack(side="left")
        mode = "CoP saja" if lt is None else ("Lean" if lt.lean else "Target")
        ttk.Label(top, text=f"   {mode}" + ("" if lt is None else f" - {len(lt.names)} arah - {a.reps}x"), style="Sub.TLabel").pack(side="left")
        ttk.Button(top, text="Pengaturan", command=self.app.back_to_setup).pack(side="right")
        # left column
        left = ttk.Frame(self, width=230); left.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        left.pack_propagate(False)
        box = ttk.LabelFrame(left, text="Parameter", padding=10); box.pack(fill="x")
        rows = [("Tinggi", f"{self.meta['height_cm']:g} cm" if self.meta["height_cm"] else "-"), ("Tangan", self.meta["upper"]),
                ("Kaki", self.meta["stance"]), ("Alas kaki", self.meta["footwear"])]
        if lt is not None:
            rows += [("Waktu condong", f"{a.lean_s:g} s") if lt.lean else ("Batas capai", f"{a.reach_timeout:g} s"),
                     ("Elips LOS", f"{2 * a.los_x:g} x {2 * a.los_y:g} cm")]
        for k, (l, v) in enumerate(rows):
            ttk.Label(box, text=l, style="Muted.TLabel").grid(row=k, column=0, sticky="w")
            ttk.Label(box, text=v, style="Card.TLabel").grid(row=k, column=1, sticky="w", padx=8)
        disp = ttk.LabelFrame(left, text="Tampilan", padding=10); disp.pack(fill="x", pady=10)
        self.v_auto, self.v_fx, self.v_fy = tk.BooleanVar(value=True), tk.BooleanVar(value=eng.flip["x"]), tk.BooleanVar(value=eng.flip["y"])
        ttk.Checkbutton(disp, text="AutoScale", variable=self.v_auto, command=lambda: self.view.update(auto=self.v_auto.get())).pack(anchor="w")
        ttk.Checkbutton(disp, text="Cermin kiri-kanan", variable=self.v_fx, command=lambda: eng.flip.update(x=self.v_fx.get())).pack(anchor="w")
        ttk.Checkbutton(disp, text="Cermin depan-belakang", variable=self.v_fy, command=lambda: eng.flip.update(y=self.v_fy.get())).pack(anchor="w")
        z = ttk.Frame(disp, style="Card.TFrame"); z.pack(fill="x", pady=(6, 0))
        ttk.Button(z, text="Zoom +", style="Small.TButton", command=lambda: self.zoom(1 / 1.25)).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(z, text="Zoom -", style="Small.TButton", command=lambda: self.zoom(1.25)).pack(side="left", expand=True, fill="x")
        sig = ttk.LabelFrame(left, text="Sinyal", padding=10); sig.pack(fill="x")
        self.l_read = ttk.Label(sig, text="", style="Card.TLabel", font=("Consolas" if sys.platform.startswith("win") else "DejaVu Sans Mono", 9), justify="left")
        self.l_read.pack(anchor="w")
        # centre: banner + tabs
        centre = ttk.Frame(self); centre.grid(row=1, column=1, sticky="nsew")
        centre.rowconfigure(1, weight=1); centre.columnconfigure(0, weight=1)
        self.banner = tk.Label(centre, text="", bg="#dde3ea", fg=INK, font=(self.app.fam, 18, "bold"), pady=10)
        self.banner.grid(row=0, column=0, sticky="we", pady=(0, 8))
        self.nb = ttk.Notebook(centre); self.nb.grid(row=1, column=0, sticky="nsew")
        t1, t2, t3 = ttk.Frame(self.nb), ttk.Frame(self.nb), ttk.Frame(self.nb)
        t4 = ttk.Frame(self.nb)
        self.nb.add(t1, text="CoP"); self.nb.add(t2, text="Sinyal"); self.nb.add(t3, text="Tabel hasil"); self.nb.add(t4, text="Grafik CoP")
        self.fig3 = Figure(figsize=(7, 8), dpi=100, facecolor="white")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=t4); self.canvas3.get_tk_widget().pack(fill="both", expand=True)
        self.fig = Figure(figsize=(6, 6), dpi=100, facecolor=CARD); self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=t1); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.setup_axes()
        self.fig2 = Figure(figsize=(6, 4), dpi=100, facecolor=CARD); self.ax2 = self.fig2.add_subplot(111)
        self.fig2.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.12)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=t2); self.canvas2.get_tk_widget().pack(fill="both", expand=True)
        self.l_tot, = self.ax2.plot([], [], c="#1b2733", lw=1.6, label="Total")
        self.l_l, = self.ax2.plot([], [], c="#1f77b4", lw=1.2, label="Plate kiri")
        self.l_r, = self.ax2.plot([], [], c="#d95f02", lw=1.2, label="Plate kanan")
        self.ax2.set_xlabel("detik"); self.ax2.set_ylabel("Gaya vertikal (N)"); self.ax2.grid(alpha=.3); self.ax2.legend(loc="upper left")
        if lt is None or lt.lean:
            cols = [("Reaction time (ms)", "reaction_ms", "{:.0f}"), ("Velocity (m/s)", "velocity_m_s", "{:.3f}"),
                    ("Excursion (m)", "max_excursion_m", "{:.3f}"), ("Dir. control (%)", "direction_control_pct", "{:.1f}")]
            if a.height_m: cols.append(("Angle (deg)", "angle_deg", "{:.2f}"))
        else:
            cols = [("Reaction (s)", "reaction_s", "{:.2f}"), ("To target (s)", "time_to_target_s", "{:.2f}"), ("Return (s)", "return_s", "{:.2f}"),
                    ("Cycle (s)", "cycle_s", "{:.2f}"), ("Max exc. (%)", "max_excursion_pct", "{:.0f}")]
        self.cols = cols; self._nrows = -1
        self.tree = ttk.Treeview(t3, columns=[c[1] for c in cols], show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Arah / trial"); self.tree.column("#0", width=200, anchor="w")
        for lab, key, _ in cols: self.tree.heading(key, text=lab); self.tree.column(key, width=120, anchor="e")
        self.tree.tag_configure("dir", background="#dbe7f3", font=(self.app.fam, 10, "bold")); self.tree.tag_configure("bad", foreground=BAD)
        sb = ttk.Scrollbar(t3, orient="vertical", command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.tree.pack(fill="both", expand=True)
        # right column
        right = ttk.Frame(self, width=190); right.grid(row=1, column=2, sticky="ns", padx=(10, 0)); right.pack_propagate(False)
        tl = ttk.LabelFrame(right, text="Time Left (s)", padding=10); tl.pack(fill="x")
        self.l_time = ttk.Label(tl, text="0", font=(self.app.fam, 34, "bold"), foreground=ACCENT, style="Card.TLabel"); self.l_time.pack()
        self.l_trial = ttk.Label(tl, text="", style="Muted.TLabel"); self.l_trial.pack()
        rb = ttk.LabelFrame(right, text="Hasil", padding=10); rb.pack(fill="x", pady=10)
        self.b_copy = ttk.Button(rb, text="Salin tabel", command=self.copy_table, state="disabled"); self.b_copy.pack(fill="x", pady=2)
        self.b_pdf = ttk.Button(rb, text="Buka PDF", command=lambda: open_file(self.last_pdf), state="disabled"); self.b_pdf.pack(fill="x", pady=2)
        self.b_dir = ttk.Button(rb, text="Buka folder", command=lambda: open_file(a.outdir)); self.b_dir.pack(fill="x", pady=2)
        # bottom
        bot = ttk.Frame(self); bot.grid(row=2, column=0, columnspan=3, sticky="we", pady=(10, 0))
        self.b_tare = ttk.Button(bot, text="Tare (plate kosong)", command=self.do_tare); self.b_tare.pack(side="left", padx=(0, 6))
        self.b_centre = ttk.Button(bot, text="Set tengah", command=self.set_centre); self.b_centre.pack(side="left", padx=6)
        self.b_start = ttk.Button(bot, text="START", style="Accent.TButton", command=self.start); self.b_start.pack(side="left", padx=6)
        self.b_stop = ttk.Button(bot, text="STOP & SIMPAN", style="Stop.TButton", command=self.stop, state="disabled"); self.b_stop.pack(side="left", padx=6)
        self.b_cancel = ttk.Button(bot, text="Batal (tanpa simpan)", command=self.cancel, state="disabled"); self.b_cancel.pack(side="left", padx=6)
        self.status = ttk.Label(bot, text="", style="Sub.TLabel"); self.status.pack(side="right")
        self.app.bind("<space>", lambda e: (self.stop() if self.eng.meas_on else self.start()) if isinstance(self.app.page, MeasurePage) and not isinstance(self.app.focus_get(), (ttk.Entry, tk.Text, ttk.Combobox)) else None)

    def setup_axes(self):
        a, ax = self.a, self.ax
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.25); ax.tick_params(labelsize=9)
        ax.set_xlabel("ML (cm)"); ax.set_ylabel("AP (cm)")
        ax.add_patch(Ellipse((0, 0), 2 * a.los_x, 2 * a.los_y, fc="#fdecea", ec="#d9534f", lw=2, alpha=.6, zorder=1))
        ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
        lt = self.eng.lt
        self.home = Circle((0, 0), a.home_r, fill=False, ls="--", ec="0.45", lw=1.6, visible=lt is not None, zorder=2); ax.add_patch(self.home)
        self.target = Circle((0, 0), a.target_r, fc="#f39c12", ec="k", visible=False, zorder=4); ax.add_patch(self.target)
        self.cues = {}
        if lt is not None and lt.lean:
            pos = {n: lt.target_xy(n) for n in lt.names}; pts = list(pos.values())
            dmin = min([np.hypot(p[0] - q[0], p[1] - q[1]) for i, p in enumerate(pts) for q in pts[i + 1:]] or [1e9])
            r = min(a.target_r * 1.5, 0.45 * dmin)
            for n, (cx, cy) in pos.items():
                self.cues[n] = Circle((cx, cy), r, fc="#d5d9de", ec="#8f98a3", lw=1.2, zorder=3); ax.add_patch(self.cues[n])
        self.trail_lc = LineCollection([], linewidths=2.2, zorder=5); ax.add_collection(self.trail_lc)
        self.dot, = ax.plot([], [], "o", c=BLUE, ms=15, mec="white", mew=2, zorder=6)
        for txt, xy, ha, va in (("DEPAN", (.5, .985), "center", "top"), ("BELAKANG", (.5, .015), "center", "bottom"),
                                ("KIRI", (.012, .5), "left", "center"), ("KANAN", (.988, .5), "right", "center")):
            ax.text(*xy, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=9, color=MUTED, weight="bold", zorder=7)
        R = self.view["R"]; ax.set_xlim(-R, R); ax.set_ylim(-R, R)

    # ---- actions
    def zoom(self, f): self.view.update(R=self.view["R"] * f, auto=False); self.v_auto.set(False)
    def do_tare(self):
        if self.eng.meas_on: return
        self.eng.retare()
    def set_centre(self):
        if self.eng.recent: self.eng.centre = np.mean(self.eng.recent, axis=0)

    def start(self):
        eng, a, lt = self.eng, self.a, self.eng.lt
        if eng.meas_on or eng.tare["left"] > 0: return
        if lt is not None:
            if len(eng.recent) < int(0.5 * a.fs): messagebox.showinfo("Belum siap", "Subjek harus sudah berdiri di plate dan diam."); return
            eng.centre = np.mean(list(eng.recent)[-int(a.fs):], axis=0)      # home = quiet standing of the last second
            lt.begin(); lt.t_rec0 = eng.t_now
        self.finished = False
        self.base = os.path.join(a.outdir, (f"{a.subject}_" if a.subject else "") + f"{datetime.datetime.now():%Y%m%d_%H%M%S}")
        self.meta["when"] = datetime.datetime.now()
        try:
            with open(self.base + "_params.json", "w", encoding="utf-8") as f:
                json.dump({"app": VERSION, "meta": {k: str(v) for k, v in self.meta.items()}, "args": vars(a)}, f, indent=1, ensure_ascii=False, default=str)
        except Exception as e: print("params file failed:", e)
        eng.rec.clear(); eng.meas_on, eng.n0 = True, eng.cnt
        self.t_wall = time.time()
        self.b_start.configure(state="disabled"); self.b_stop.configure(state="normal"); self.b_cancel.configure(state="normal")
        self.b_tare.configure(state="disabled"); self.b_pdf.configure(state="disabled"); self.nb.select(0)

    def cancel(self):
        eng = self.eng
        if not eng.meas_on: return
        if not messagebox.askyesno("Batalkan", "Pengukuran dibatalkan tanpa menyimpan?"): return
        eng.meas_on = False; eng.rec.clear()
        if eng.lt: eng.lt.stop()
        self.idle_buttons()

    def idle_buttons(self):
        self.b_start.configure(state="normal"); self.b_stop.configure(state="disabled"); self.b_cancel.configure(state="disabled")
        self.b_tare.configure(state="normal")

    def stop(self): self.finalize()

    def finalize(self):
        eng, a, lt = self.eng, self.a, self.eng.lt
        if not eng.meas_on: return
        eng.meas_on = False
        self.banner.configure(text="Menyimpan...", bg="#dde3ea", fg=INK); self.update_idletasks()
        saved = []
        try:
            if eng.rec:
                np.savetxt(self.base + "_raw.csv", np.vstack(eng.rec), delimiter=",", header=L.CSV_HEADER, comments="", fmt="%.5f"); saved.append("raw.csv")
            if lt is not None:
                done = lt.state == "DONE"; lt.stop()
                if lt.rows:
                    lt.save_csv(self.base + "_trials.csv"); saved.append("trials.csv")
                    if lt.lean: lt.save_summary_csv(self.base + "_summary.csv"); saved.append("summary.csv")
                    self.meta.setdefault("when", datetime.datetime.now())
                    S = self.session_data()
                    self.last_pdf = los_report.build_report(self.base + "_report.pdf", a, self.meta, lt, S); saved.append("report.pdf")
                    self.show_session(S)
                    self.fill_tree(); self._nrows = len(lt.rows); self.nb.select(2)
                    self.b_pdf.configure(state="normal"); self.b_copy.configure(state="normal")
                self.finished = done
        except Exception as e:
            messagebox.showerror("Gagal menyimpan", f"{type(e).__name__}: {e}")
        self.idle_buttons()
        self.status.configure(text=("Tersimpan: " + ", ".join(saved)) if saved else "Tidak ada data yang disimpan")

    def session_data(self):
        """Whole recording in the screen view (cm, home = 0), decimated to ~100 Hz."""
        eng, a = self.eng, self.a
        if not eng.rec: return None
        arr = np.vstack(eng.rec)[::max(1, int(a.fs // 100))]
        sx = (arr[:, 12] - eng.centre[0]) * 100.0 * (-1 if eng.flip["x"] else 1)
        sy = (arr[:, 13] - eng.centre[1]) * 100.0 * (-1 if eng.flip["y"] else 1)
        return dict(t=arr[:, 0], sx=sx, sy=sy)

    def show_session(self, S):
        lt = self.eng.lt
        if S is None or lt is None or not lt.rows: return
        self.fig3.clf(); los_report.draw_session(self.fig3, (0.0, 0.0, 1.0, 1.0), self.a, lt, S, fs=9, layout="side"); self.canvas3.draw_idle()

    def fill_tree(self):
        lt = self.eng.lt
        self.tree.delete(*self.tree.get_children())
        for d in lt.names:
            rs = [r for r in lt.rows if r["direction"] == d]
            if not rs: continue
            ok = [r for r in rs if r["outcome"] == "ok"]
            def mean(k):
                v = [r[k] for r in ok if r[k] == r[k]]; return float(np.mean(v)) if v else float("nan")
            fm = lambda x, f: "-" if x != x else f.format(x)
            name = L.EN[d] if lt.lean else d
            p = self.tree.insert("", "end", text=f"{name}  (mean, n={len(ok)})", values=[fm(mean(k), f) for _, k, f in self.cols], tags=("dir",), open=True)
            for r in rs:
                self.tree.insert(p, "end", text=f"   T{r['trial']}" + ("" if r["outcome"] == "ok" else f"  [{r['outcome']}]"),
                                 values=[fm(r[k], f) for _, k, f in self.cols], tags=() if r["outcome"] == "ok" else ("bad",))

    def copy_table(self):
        lt = self.eng.lt
        if lt is None or not lt.rows: return
        lines = ["\t".join(["Direction", "Trial"] + [c[0] for c in self.cols])]
        for r in lt.rows:
            lines.append("\t".join([L.EN[r["direction"]] if lt.lean else r["direction"], str(r["trial"])] +
                                   ["-" if r[k] != r[k] else f.format(r[k]) for _, k, f in self.cols]))
        self.clipboard_clear(); self.clipboard_append("\n".join(lines))

    # ---- periodic update
    def tick(self):
        if not self.alive: return
        try:
            self.eng.poll()
            if self.eng.lt is not None and self.eng.lt.state == "DONE" and self.eng.meas_on: self.finalize()
            self.refresh()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.alive = False; messagebox.showerror("Kesalahan", f"{type(e).__name__}: {e}\n\nLihat jendela terminal untuk detail.")
            return
        self.after(33, self.tick)

    def banner_set(self, text, bg, fg="white"): self.banner.configure(text=text, bg=bg, fg=fg)

    def refresh(self):
        eng, a, lt = self.eng, self.a, self.eng.lt
        # banner
        if eng.tare["left"] > 0: self.banner_set(f"TARE... kosongkan plate ({eng.tare['left'] / a.fs:.1f} s)", WARN)
        elif eng.warn: self.banner_set("Gaya NEGATIF di salah satu plate: cek mode terminal / tare", BAD)
        elif eng.meas_on and lt is not None:
            m = {"HOME_WAIT": ("Berdiri tenang di tengah (lingkaran putus-putus)", "#4a6785"), "ARMED": ("Tunggu... jangan bergerak", "#4a6785"),
                 "SHOW": ("", "#4a6785"), "TARGET": (("CONDONG KE LAMPU SEBISA MUNGKIN" if lt.lean else "GERAK KE TARGET"), WARN),
                 "RETURN": ("KEMBALI KE TENGAH", OK), "DONE": ("Selesai", OK)}
            t, c = m.get(lt.state, ("", "#4a6785")); self.banner_set(t or "...", c)
        elif eng.meas_on: self.banner_set(f"MEREKAM {time.time() - self.t_wall:.1f} s", BAD)
        elif self.finished: self.banner_set("Selesai. Hasil dan laporan PDF tersimpan.", OK)
        else: self.banner_set("PREVIEW: subjek naik ke plate, lalu START" if eng.tare_s else "PREVIEW: tekan START", "#4a6785")
        # time left / trial
        tl = 0.0
        if lt is not None and eng.meas_on and lt.cur is not None and lt.state == "TARGET":
            tl = max(0.0, (a.lean_s if lt.lean else a.reach_timeout) - (eng.t_now - lt.cur["t0"]))
        self.l_time.configure(text=f"{tl:.0f}")
        self.l_trial.configure(text=(f"Trial {min(len(lt.rows) + (1 if lt.cur else 0), lt.total)}/{lt.total}" if lt is not None and lt.total else ""))
        if lt is not None and len(lt.rows) != self._nrows and eng.meas_on:      # table grows trial by trial
            self._nrows = len(lt.rows); self.fill_tree(); self.b_copy.configure(state="normal")
        # lights
        if lt is not None:
            if lt.lean:
                for n, p in self.cues.items():
                    on = lt.target_visible and lt.cur is not None and lt.cur["name"] == n
                    p.set_facecolor("#f39c12" if on else "#d5d9de")
            else:
                self.target.set_visible(lt.target_visible)
                if lt.cur is not None: self.target.center = (lt.cur["tx"], lt.cur["ty"])
            ret = lt.state == "RETURN"
            self.home.set_edgecolor(OK if ret else "0.45"); self.home.set_linewidth(4 if ret else 1.6)
        # readouts
        if np.isfinite(eng.fz):
            tot = max(eng.lfz + eng.rfz, 1e-9)
            self.l_read.configure(text=f"Fz   {eng.fz:6.0f} N\nKiri {100 * eng.lfz / tot:5.0f} %\nKanan{100 * eng.rfz / tot:5.0f} %\nDitolak {eng.dropped}")
        # plot
        if self.nb.index(self.nb.select()) == 0:
            if eng.trail and eng.valid:
                xy = eng.screen_xy()
                if self.view["auto"]:
                    need = max(self.R0, np.nanmax(np.abs(xy)) * 1.15)
                    self.view["R"] = need if need > self.view["R"] else max(need, self.view["R"] * 0.98)
                P = xy[::max(1, len(xy) // 300)]
                if len(P) > 1:
                    seg = np.stack([P[:-1], P[1:]], axis=1); al = np.linspace(0.05, 0.85, len(seg))
                    col = np.zeros((len(seg), 4)); col[:, :3] = matplotlib.colors.to_rgb(BLUE); col[:, 3] = al
                    self.trail_lc.set_segments(seg); self.trail_lc.set_color(col)
                self.dot.set_data([xy[-1, 0]], [xy[-1, 1]])
                self.dot.set_color(OK if np.hypot(xy[-1, 0] / a.los_x, xy[-1, 1] / a.los_y) >= 1 else BLUE)
            else:
                self.dot.set_data([], []); self.trail_lc.set_segments([])
            R = self.view["R"]
            if abs(self.ax.get_xlim()[1] - R) > 1e-9: self.ax.set_xlim(-R, R); self.ax.set_ylim(-R, R)
            self.canvas.draw_idle()
        elif self.nb.index(self.nb.select()) == 1 and eng.sig:
            S = np.array(eng.sig); t = S[:, 0] - S[-1, 0]
            self.l_tot.set_data(t, S[:, 1]); self.l_l.set_data(t, S[:, 2]); self.l_r.set_data(t, S[:, 3])
            self.ax2.set_xlim(t[0] if t[0] < -1 else -9, 0); self.ax2.set_ylim(min(-20, S[:, 2:].min() - 20), max(100, S[:, 1].max() * 1.15))
            self.canvas2.draw_idle()

    def shutdown(self):
        self.alive = False
        try:
            if self.eng.meas_on: self.finalize()
        except Exception: pass
        try: self.src.stop()
        except Exception: pass
        try: self.app.unbind("<space>")
        except Exception: pass


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
