"""Tkinter GUI for the MLB prediction engine.

Dropdown-driven input: teams, date, stadium, starters, two ordered 9-man
lineups, day/night, and weather. Outputs per run: every batter's calibrated
HR probability (with fair odds), hit probability, both starters' projected
strikeouts with over-probabilities, and game totals.

Run:
    python "Tools/3) GUI.py"
"""

import base64
import datetime as dt
import io
import queue
import re
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

import sys
# the prediction engine (predict.py and friends) lives in Model/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model"))

# Silence known-benign pandas noise from the prediction engine (mixed-type
# CSV column inference and fragmented-frame inserts).
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
# MLB logo, embedded as a base64-encoded PNG (228x128 RGBA) so the GUI
# needs no image file on disk. To regenerate from a source image: open
# it with Pillow, .convert('RGBA'), resize to height 128 (LANCZOS),
# save as optimized PNG, and base64-encode the bytes.
LOGO_B64 = (
    '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'
)

# MLB brand colors
NAVY = "#041E42"
RED = "#D50032"
WHITE = "#FFFFFF"
STRIPE = "#EAF0F8"      # light row stripe
TOPPICK = "#FBE3E9"     # light red tint for top HR picks
DISABLED = "#5A6B84"


def load_logo(height=64):
    """MLB logo as a tk PhotoImage, decoded from the embedded base64 PNG
    (Pillow for smooth scaling if present)."""
    try:
        from PIL import Image, ImageTk
        img = Image.open(io.BytesIO(base64.b64decode(LOGO_B64)))
        w = int(img.width * height / img.height)
        return ImageTk.PhotoImage(img.resize((w, height), Image.LANCZOS))
    except Exception:
        try:
            # Tk 8.6+ decodes base64 PNG data directly
            img = tk.PhotoImage(data=LOGO_B64)
            return img.subsample(max(1, img.height() // height))
        except Exception:
            return None


class App(tk.Tk):
    # opening size — wide enough for the full top form row through the HP
    # Umpire box (col 10). _fit_minsize() raises it if the built form needs more.
    START_W, START_H = 1220, 920

    def __init__(self):
        super().__init__()
        self.title("MLB Prediction Engine")
        self.geometry(f"{self.START_W}x{self.START_H}")
        self._apply_style()
        self.pred = None
        self.pools = {}      # abbrev -> dict(batters={label: pid}, pitchers={...})
        self.abbrev_full = {}
        # the worker thread only writes these; the main thread polls them
        # (tkinter's after() must never be called from a worker thread)
        self._load_state = None
        self._load_msg = "starting..."
        self._pred_state = None
        # ONE persistent worker runs every model job. XGBoost/LightGBM/
        # CUDA inference leaves thread-local native state whose teardown
        # fail-fasts the process (0xC0000409) when the thread that ran
        # it exits mid-session — a thread-per-job design crashed the app
        # right after each prediction finished. This thread never exits.
        self._jobs = queue.Queue()
        threading.Thread(target=self._worker_loop, daemon=True).start()
        self._build_layout()
        self._fit_minsize()
        self.status.set("Loading data and models...")
        self._jobs.put(self._load)
        self.after(200, self._poll_load)

    def _worker_loop(self):
        while True:
            self._jobs.get()()

    # ------------------------------------------------------------ setup

    def _fit_minsize(self):
        """The window itself no longer scrolls, so it must never be shrinkable
        past what the fixed form needs — otherwise content would be silently
        clipped with no scrollbar to reach it. Measure the built layout once and
        pin that as the floor (clamped to the screen, so a small display still
        gets a usable window). Everything above the floor goes to the slate, the
        one elastic region, which scrolls internally."""
        self.update_idletasks()
        need_w = min(self.winfo_reqwidth(), self.winfo_screenwidth())
        need_h = min(self.winfo_reqheight(), self.winfo_screenheight())
        self.minsize(need_w, need_h)
        self.geometry(f"{max(self.START_W, need_w)}x"
                      f"{max(self.START_H, need_h)}")

    def _load(self):
        try:
            from predict import Predictor

            def tick(msg):
                self._load_msg = msg
            self.pred = Predictor(progress=tick)
            self._load_msg = "building player pools..."
            self._build_pools()
            self._load_state = ("ok", None)
        except Exception as e:
            self._load_state = ("err", str(e))

    def _poll_load(self):
        if self._load_state is None:
            self.status.set(f"Loading: {self._load_msg}")
            self.after(200, self._poll_load)
            return
        state, err = self._load_state
        if state == "ok":
            self._on_ready()
        else:
            self.status.set(f"LOAD FAILED: {err}")
            messagebox.showerror("Load failed", err)

    def _build_pools(self):
        season = int(self.pred.stores.raw["games"]["Season"].max())
        bs = pd.read_csv(DATA_DIR / "mlb_batting_stats.csv",
                         encoding="utf-8-sig", usecols=["Year", "Team", "TeamName"])
        pairs = bs[bs["Year"] == bs["Year"].max()].drop_duplicates()
        self.abbrev_full = dict(zip(pairs["Team"], pairs["TeamName"]))
        full_abbrev = {v: k for k, v in self.abbrev_full.items()}

        ros = self.pred.stores.raw["rosters"]
        for _, r in ros.iterrows():
            ab = full_abbrev.get(r["Team"])
            if ab is None:
                continue
            pool = self.pools.setdefault(ab, {"batters": {}, "pitchers": {}})
            label = f'{r["Name"]} ({r["Position"]})'
            if r["Position"] in ("Rotation", "Bullpen"):
                pool["pitchers"][label] = r["PlayerId"]
            else:
                pool["batters"][label] = r["PlayerId"]

        # Depth-chart rosters lag trades and call-ups (e.g. a player dealt
        # mid-season). Anyone who actually appeared in current-season game
        # logs is added to the pool of the team they last played for.
        gb = self.pred.stores.raw["gb"]
        cur = gb[gb["Season"] == season].sort_values("Date")
        for pid, r in cur.groupby("PlayerId").last().iterrows():
            pool = self.pools.setdefault(r["Team"], {"batters": {}, "pitchers": {}})
            if pid in pool["batters"].values() or pid in pool["pitchers"].values():
                continue
            label = f'{r["Name"]} ({r["Position"] or "?"})'
            if label in pool["batters"]:
                label = f'{r["Name"]} [{pid}]'
            pool["batters"][label] = pid
        gp = self.pred.stores.raw["gp"]
        curp = gp[gp["Season"] == season].sort_values("Date")
        agg = curp.groupby("PlayerId").agg(
            n=("GamePk", "size"), gs=("GS", "sum"))
        for pid, r in curp.groupby("PlayerId").last().iterrows():
            # skip position players with a mop-up appearance or two
            if agg.loc[pid, "n"] < 3 and agg.loc[pid, "gs"] == 0:
                continue
            pool = self.pools.setdefault(r["Team"], {"batters": {}, "pitchers": {}})
            if pid in pool["pitchers"].values():
                continue
            label = f'{r["Name"]} (P)'
            if label in pool["pitchers"]:
                label = f'{r["Name"]} [{pid}]'
            pool["pitchers"][label] = pid

        games = self.pred.stores.raw["games"]
        parks = self.pred.stores.raw["parks"]
        self.venues = sorted(set(parks["Ballpark"]) |
                             set(games.loc[games["Season"] == season, "Venue"]))
        self.wind_dirs = sorted(games["WindDir"].dropna().unique())
        self.conditions = sorted(games["Condition"].dropna().unique())
        # HP-umpire name -> HpUmpId, for the form's editable umpire field.
        umps = self.pred.stores.raw.get("umps")
        self.ump_name_to_id = {}
        if umps is not None:
            u = umps.dropna(subset=["HpUmp", "HpUmpId"])
            self.ump_name_to_id = {n: int(i) for n, i in
                                   zip(u["HpUmp"], u["HpUmpId"])}
        self.ump_names = sorted(self.ump_name_to_id)
        # home team -> default venue
        self.team_park = {full_abbrev.get(t): b for b, t in
                          zip(parks["Ballpark"], parks["Team"]) if full_abbrev.get(t)}

    def _on_ready(self):
        self._refresh_team_options()
        self.cb_venue["values"] = self.venues
        self.cb_wdir["values"] = self.wind_dirs
        self.cb_cond["values"] = self.conditions
        self.cb_ump["values"] = self.ump_names
        self.btn_predict["state"] = "normal"
        self.status.set("Ready. Pick teams, fill lineups (or auto-fill), Predict.")
        self._load_todays_file(silent=True)
        self._health_check()

    def _health_check(self):
        """Warn when predictions would be built on bad inputs: the morning
        data job failed (Scrapers/update_all.py writes its outcome to
        Logs/last_run_status.json) or the game logs have gone stale
        mid-season. Without this, the only failure signal is a log line."""
        import json
        problems = []
        status_file = DATA_DIR.parent / "Logs" / "last_run_status.json"
        try:
            status = json.loads(status_file.read_text())
            if not status.get("ok"):
                jobs = ", ".join(status.get("failed_jobs", [])) or "unknown"
                problems.append(
                    f"The last data update FAILED (finished "
                    f"{status.get('finished', '?')}; failed: {jobs}).\n"
                    f"Data was restored from backups and the retrain was "
                    f"skipped — see the newest Logs/update_*.log.")
        except (OSError, ValueError):
            pass                    # no status yet: job hasn't run since setup
        try:
            games = self.pred.stores.raw["games"]
            newest = pd.to_datetime(games["Date"]).max()
            age = (pd.Timestamp.today().normalize() - newest.normalize()).days
            if 5 <= dt.date.today().month <= 9 and age > 6:
                problems.append(
                    f"Newest game in the data is {newest.date()} "
                    f"({age} days ago) — mid-season that means the daily "
                    f"update is not ingesting new games. Predictions will "
                    f"use stale form/rosters.")
        except Exception:           # noqa: BLE001 — health check never blocks
            pass
        if problems:
            messagebox.showwarning("Data health", "\n\n".join(problems))

    def _load_todays_file(self, silent=False):
        """Populate the slate from Data/todays_games.json (written by
        "Tools/1) Get Todays Games.py") if it exists."""
        path = DATA_DIR / "todays_games.json"
        if not path.exists():
            if not silent:
                messagebox.showinfo(
                    "No file", "Data/todays_games.json not found.\nRun: "
                    "python \"Tools/1) Get Todays Games.py\"")
            return
        self._load_slate_file(path, silent=silent)

    def _load_slate_from_file(self):
        """Pick slate JSONs (todays_games.json shape) — archived or
        regenerated slates from Data/slates. One file loads into the
        slate for editing; several batch-predict back-to-back, each
        producing its own workbook. The served workbook is named from
        the slate's own date, so old slates file themselves under the
        right day in Predictions/."""
        slates_dir = DATA_DIR / "slates"
        paths = filedialog.askopenfilenames(
            title="Load slate JSON(s) — select several to batch-predict",
            initialdir=str(slates_dir if slates_dir.is_dir() else DATA_DIR),
            filetypes=[("Slate JSON", "*.json"), ("All files", "*.*")])
        if not paths:
            return
        if len(paths) == 1:
            self._load_slate_file(Path(paths[0]))
        else:
            self._batch_predict([Path(p) for p in paths])

    @staticmethod
    def _read_slate_specs(path):
        """Parse a slate JSON into predict-ready specs: earliest first
        pitch first (the workbook sheets keep this order), lineups
        tuple-ified. Raises on an unreadable file; the returned list is
        empty when the file has no games."""
        import json
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        specs = payload.get("games", [])
        specs.sort(key=lambda s: s.get("start_et") or "99:99")
        for spec in specs:
            spec["away_lineup"] = [tuple(x) for x in spec.get("away_lineup", [])]
            spec["home_lineup"] = [tuple(x) for x in spec.get("home_lineup", [])]
        return specs, payload

    def _load_slate_file(self, path, silent=False):
        try:
            specs, payload = self._read_slate_specs(path)
        except Exception as e:
            if not silent:
                messagebox.showerror("Load failed", str(e))
            return
        if not specs:
            if not silent:
                messagebox.showinfo("Empty slate",
                                    f"{Path(path).name} has no games.")
            return
        self._clear_slate()
        for spec in specs:
            self.slate.append(spec)
            self.lb_slate.insert("end", self._slate_row_text(spec))
        scraped = str(payload.get("scraped_at", ""))[:16].replace("T", " ")
        date = str(specs[0].get("date", ""))
        self.status.set(f"Loaded {len(specs)} games ({date}) from "
                        f"{Path(path).name}"
                        f"{f' (scraped {scraped})' if scraped else ''}. "
                        f"Click a game to load/edit it")

    # ----------------------------------------------------------- layout

    def _apply_style(self):
        self.configure(bg=NAVY)
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=NAVY, foreground=WHITE,
                    font=("Segoe UI", 10))
        s.configure("Title.TLabel", font=("Segoe UI", 17, "bold"),
                    foreground=WHITE)
        s.configure("Sub.TLabel", foreground="#9FB3D1")
        s.configure("TLabelframe", background=NAVY, bordercolor=RED,
                    relief="solid")
        s.configure("TLabelframe.Label", background=NAVY, foreground=WHITE,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TButton", background=RED, foreground=WHITE,
                    font=("Segoe UI", 10, "bold"), padding=(10, 5),
                    bordercolor=RED, focuscolor=RED)
        s.map("TButton",
              background=[("disabled", DISABLED), ("active", "#F0234F")],
              foreground=[("disabled", "#C4CDD9")])
        for w in ("TCombobox", "TSpinbox", "TEntry"):
            s.configure(w, fieldbackground=WHITE, foreground=NAVY,
                        bordercolor="#7A8CA8", arrowcolor=NAVY,
                        insertcolor=NAVY)
        s.map("TCombobox",
              fieldbackground=[("readonly", WHITE)],
              foreground=[("readonly", NAVY)])
        s.configure("Treeview", background=WHITE, fieldbackground=WHITE,
                    foreground=NAVY, rowheight=24, font=("Segoe UI", 10))
        s.configure("Treeview.Heading", background=RED, foreground=WHITE,
                    font=("Segoe UI", 10, "bold"), relief="flat")
        s.map("Treeview.Heading", background=[("active", "#F0234F")])
        self.option_add("*TCombobox*Listbox.background", WHITE)
        self.option_add("*TCombobox*Listbox.foreground", NAVY)
        self.option_add("*TCombobox*Listbox.selectBackground", RED)
        self.option_add("*TCombobox*Listbox.selectForeground", WHITE)

    @staticmethod
    def _tree_with_scroll(parent, **kw):
        """A Treeview paired with a vertical scrollbar in its own frame."""
        frame = tk.Frame(parent, bg=NAVY)
        tv = ttk.Treeview(frame, **kw)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tv.pack(side="left", fill="both", expand=True)
        return frame, tv

    def _build_layout(self):
        header = tk.Frame(self, bg=NAVY)
        header.pack(fill="x", padx=8, pady=(10, 2))
        self._logo = load_logo(height=64)
        if self._logo is not None:
            tk.Label(header, image=self._logo, bg=NAVY).pack(side="left",
                                                             padx=(4, 14))
        titles = tk.Frame(header, bg=NAVY)
        titles.pack(side="left")
        ttk.Label(titles, text="MLB Prediction Engine",
                  style="Title.TLabel").pack(anchor="w")

        # fixed bottom bar first, so it stays pinned below the body
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=6)
        self.btn_predict = ttk.Button(bottom, text="Predict", state="disabled",
                                      command=self._predict_clicked)
        self.btn_predict.pack(side="left")
        self.status = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status, wraplength=820,
                  justify="left").pack(side="left", padx=12)

        # The window itself does NOT scroll — the form is always fully visible
        # and the Predict bar stays pinned. The SLATE is the only scrolling
        # region (its listbox owns the one scrollbar): it is also the only part
        # that grows without bound, so it absorbs the spare vertical space and
        # scrolls internally once the games outrun it. _fit_minsize() below
        # then forbids shrinking the window past the fixed form, which is what
        # makes dropping the global scrollbar safe (nothing can be clipped).
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        top = ttk.LabelFrame(body, text="Game")
        top.pack(fill="x", padx=8, pady=6)

        # Two stacked input rows (game identity, then conditions) so the
        # window stays narrow now that weather has more fields.
        def add(col, text, widget, row=0):
            ttk.Label(top, text=text).grid(row=row * 2, column=col,
                                           sticky="w", padx=4)
            widget.grid(row=row * 2 + 1, column=col, sticky="w", padx=4,
                        pady=2)
            return widget

        self.cb_away = add(0, "Away team", ttk.Combobox(top, width=6, state="readonly"))
        self.cb_home = add(1, "Home team", ttk.Combobox(top, width=6, state="readonly"))
        self.e_date = add(2, "Date (YYYY-MM-DD)", ttk.Entry(top, width=12))
        self.e_date.insert(0, dt.date.today().isoformat())
        # Optional — mlb.com drops the time once a game starts, so scraped
        # slates can arrive without it; sorts and slate order fall back
        # gracefully when blank. Accepts '7:10 PM' or '19:10'.
        self.e_start = add(3, "Start ET (opt.)", ttk.Entry(top, width=10))
        self.cb_venue = add(4, "Stadium", ttk.Combobox(top, width=28))
        self.cb_dn = add(5, "Day/Night", ttk.Combobox(
            top, width=7, state="readonly", values=["day", "night"]))
        self.cb_dn.set("day")
        self.sp_temp = add(0, "Temp °F", ttk.Spinbox(top, from_=20, to=115,
                                                     width=5), row=1)
        self.sp_temp.set(72)
        self.sp_wind = add(1, "Wind mph", ttk.Spinbox(top, from_=0, to=45,
                                                      width=5), row=1)
        self.sp_wind.set(6)
        self.cb_wdir = add(2, "Wind dir", ttk.Combobox(top, width=14), row=1)
        self.cb_cond = add(3, "Condition", ttk.Combobox(top, width=14), row=1)
        # Air-density inputs (scraped from the Open-Meteo forecast by
        # "1) Get Todays Games.py"; editable). Blank = NaN, the model imputes.
        # Under a closed roof (Condition = Dome) the model swaps humidity
        # for a fixed indoor value, so this field only matters outdoors.
        self.sp_hum = add(4, "Humidity %", ttk.Spinbox(top, from_=0, to=100,
                                                       width=5), row=1)
        self.e_pres = add(5, "Pressure hPa", ttk.Entry(top, width=7), row=1)
        # Precip (mm at start hour, Open-Meteo forecast): the
        # rain-shortening signal. Blank = NaN.
        self.e_precip = add(6, "Precip mm", ttk.Entry(top, width=6), row=1)
        # Editable; leave blank for a neutral-ump prediction. Known names
        # resolve to an HpUmpId in _collect_spec; an unknown name -> no id.
        self.cb_ump = add(7, "HP Umpire", ttk.Combobox(top, width=18), row=1)
        # spread the two input rows evenly across the panel's full width
        # (equal weights share the leftover space; no `uniform`, which would
        # force every column as wide as the Stadium box and overflow)
        for i in range(8):
            top.columnconfigure(i, weight=1)

        self.cb_away.bind("<<ComboboxSelected>>", lambda e: self._team_changed("away"))
        self.cb_home.bind("<<ComboboxSelected>>", lambda e: self._team_changed("home"))

        mid = ttk.Frame(body)
        mid.pack(fill="both", expand=True, padx=8)
        self.side_widgets = {}
        for i, side in enumerate(("away", "home")):
            f = ttk.LabelFrame(mid, text=f"{side.title()} lineup")
            f.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            mid.columnconfigure(i, weight=1)
            w = {"lineup": []}
            ttk.Label(f, text="Starting pitcher").grid(row=0, column=0, sticky="w")
            w["starter"] = ttk.Combobox(f, width=34)
            w["starter"].grid(row=0, column=1, pady=2, sticky="w")
            for slot in range(1, 10):
                ttk.Label(f, text=str(slot)).grid(row=slot, column=0, sticky="w")
                cb = ttk.Combobox(f, width=34)
                cb.grid(row=slot, column=1, pady=1, sticky="w")
                cb.bind("<<ComboboxSelected>>",
                        lambda e, s=side: self._refresh_lineup_options(s))
                w["lineup"].append(cb)
            b = ttk.Button(f, text="Auto-fill from last game",
                           command=lambda s=side: self._autofill(s))
            b.grid(row=10, column=1, sticky="e", pady=4)
            self.side_widgets[side] = w

        # the ONLY scrolling region in the app: it takes every spare pixel
        # (fill both / expand) and scrolls internally once the games outgrow it
        slate_f = ttk.LabelFrame(body, text="Slate (click a game to load it "
                                            "into the form and edit; Predict "
                                            "runs them all)")
        slate_f.pack(fill="both", expand=True, padx=8, pady=4)
        self.slate = []
        self._loaded_idx = None   # slate index currently loaded into the form
        # exportselection=False: keep the row selected while the user edits
        # form fields (otherwise clicking a combobox clears the selection)
        self.lb_slate = tk.Listbox(slate_f, height=8, bg=WHITE, fg=NAVY,
                                   selectbackground=RED, exportselection=False,
                                   font=("Segoe UI", 10))
        self.lb_slate.pack(side="left", fill="both", expand=True,
                           padx=(6, 0), pady=4)
        self.lb_slate.bind("<<ListboxSelect>>", self._slate_selected)
        lb_vsb = ttk.Scrollbar(slate_f, orient="vertical",
                               command=self.lb_slate.yview)
        self.lb_slate.configure(yscrollcommand=lb_vsb.set)
        lb_vsb.pack(side="left", fill="y", pady=4)
        sb = ttk.Frame(slate_f)
        sb.pack(side="left", padx=6)
        ttk.Button(sb, text="Add game to slate",
                   command=self._add_to_slate).pack(fill="x", pady=1)
        ttk.Button(sb, text="Update selected game",
                   command=self._update_selected).pack(fill="x", pady=1)
        ttk.Button(sb, text="Remove selected",
                   command=self._remove_from_slate).pack(fill="x", pady=1)
        ttk.Button(sb, text="🡅",
                   command=lambda: self._move_slate(-1)).pack(fill="x", pady=1)
        ttk.Button(sb, text="🡇",
                   command=lambda: self._move_slate(1)).pack(fill="x", pady=1)
        ttk.Button(sb, text="Clear slate",
                   command=self._clear_slate).pack(fill="x", pady=1)
        ttk.Button(sb, text="Load today's file",
                   command=self._load_todays_file).pack(fill="x", pady=1)
        ttk.Button(sb, text="📄",
                   command=self._load_slate_from_file).pack(fill="x", pady=1)

    # ------------------------------------------------------- interaction

    def _team_changed(self, side):
        team = (self.cb_away if side == "away" else self.cb_home).get()
        pool = self.pools.get(team, {"batters": {}, "pitchers": {}})
        w = self.side_widgets[side]
        w["starter"]["values"] = sorted(pool["pitchers"])
        w["starter"].set("")
        for cb in w["lineup"]:
            cb.set("")
        self._refresh_lineup_options(side)
        self._refresh_team_options()
        if side == "home" and team in self.team_park:
            self.cb_venue.set(self.team_park[team])

    def _refresh_team_options(self):
        """Each team dropdown hides the team picked on the other side."""
        teams = sorted(self.pools)
        away, home = self.cb_away.get(), self.cb_home.get()
        self.cb_away["values"] = [t for t in teams if t != home]
        self.cb_home["values"] = [t for t in teams if t != away]

    def _refresh_lineup_options(self, side):
        """Hide already-selected players from the other lineup slots."""
        team = (self.cb_away if side == "away" else self.cb_home).get()
        pool = sorted(self.pools.get(team, {"batters": {}})["batters"])
        w = self.side_widgets[side]
        chosen = {cb.get().strip() for cb in w["lineup"] if cb.get().strip()}
        for cb in w["lineup"]:
            own = cb.get().strip()
            cb["values"] = [p for p in pool if p not in chosen or p == own]

    def _autofill(self, side):
        team = (self.cb_away if side == "away" else self.cb_home).get()
        if not team or self.pred is None:
            return
        gb = self.pred.stores.raw["gb"]
        gp = self.pred.stores.raw["gp"]
        rows = gb[(gb["Team"] == team) & gb["BattingOrder"].notna()].copy()
        rows["bo"] = pd.to_numeric(rows["BattingOrder"], errors="coerce")
        rows = rows[rows["bo"] % 100 == 0]
        if rows.empty:
            return
        last_date = rows["Date"].max()
        last = rows[rows["Date"] == last_date].sort_values("bo")
        w = self.side_widgets[side]
        pool = self.pools.get(team, {"batters": {}})
        pid_label = {v: k for k, v in pool["batters"].items()}
        for cb, (_, r) in zip(w["lineup"], last.iterrows()):
            label = pid_label.get(r["PlayerId"], f'{r["Name"]} [{r["PlayerId"]}]')
            cb.set(label)
        st = gp[(gp["Team"] == team) & (gp["GS"] == 1)].sort_values("Date")
        if len(st):
            sp = st.iloc[-1]
            plabel = {v: k for k, v in pool.get("pitchers", {}).items()}.get(
                sp["PlayerId"], f'{sp["Name"]} [{sp["PlayerId"]}]')
            w["starter"].set(plabel)
        self._refresh_lineup_options(side)
        self.status.set(f"{side} lineup auto-filled from {last_date.date()} "
                        f"(edit as needed)")

    def _resolve(self, team, label, kind):
        """Combobox label -> PlayerId (supports 'Name [id]' fallback labels)."""
        pool = self.pools.get(team, {})
        pid = pool.get(kind, {}).get(label)
        if pid is not None:
            return int(pid)
        if "[" in label and label.endswith("]"):
            return int(label.rsplit("[", 1)[1][:-1])
        raise ValueError(f"unknown player: {label!r}")

    def _label_for(self, team, pid, kind, names=None):
        """PlayerId -> combobox label, the inverse of _resolve. Players not in
        the team's pool get a 'Name [id]' fallback label (which _resolve
        parses back), using the spec's scraped names when available."""
        pid = int(pid)
        for label, p in self.pools.get(team, {}).get(kind, {}).items():
            if int(p) == pid:
                return label
        name = (names or {}).get(str(pid))
        if not name and self.pred is not None:
            name = self.pred._name(pid)
        return f'{name or pid} [{pid}]'

    def _apply_spec(self, spec):
        """Fill every form field from a game spec — the inverse of
        _collect_spec, so a slate game can be loaded, edited, and saved back
        with 'Update selected game'."""
        self.cb_away.set(spec.get("away_team") or "")
        self._team_changed("away")
        self.cb_home.set(spec.get("home_team") or "")
        self._team_changed("home")            # sets the home park default...
        self.e_date.delete(0, "end")
        self.e_date.insert(0, spec.get("date") or dt.date.today().isoformat())
        self.e_start.delete(0, "end")
        self.e_start.insert(0, spec.get("start_et") or "")
        self.cb_venue.set(spec.get("venue") or "")   # ...the spec venue wins
        self.cb_dn.set(spec.get("day_night") or "")
        for widget, v in ((self.sp_temp, spec.get("temp")),
                          (self.sp_wind, spec.get("wind_speed")),
                          (self.sp_hum, spec.get("humidity"))):
            widget.set("" if v is None else v)
        self.e_pres.delete(0, "end")
        if spec.get("pressure") is not None:
            self.e_pres.insert(0, spec["pressure"])
        self.e_precip.delete(0, "end")
        if spec.get("precip") is not None:
            self.e_precip.insert(0, spec["precip"])
        self.cb_wdir.set(spec.get("wind_dir") or "")
        self.cb_cond.set(spec.get("condition") or "")
        self.cb_ump.set(spec.get("hp_ump") or "")

        names = spec.get("names") or {}
        for side in ("away", "home"):
            team = spec.get(f"{side}_team")
            w = self.side_widgets[side]
            st = spec.get(f"{side}_starter")
            w["starter"].set(
                self._label_for(team, st, "pitchers", names) if st else "")
            for cb in w["lineup"]:
                cb.set("")
            for pid, slot in spec.get(f"{side}_lineup", []):
                if 1 <= int(slot) <= 9:
                    w["lineup"][int(slot) - 1].set(
                        self._label_for(team, pid, "batters", names))
            self._refresh_lineup_options(side)

    def _collect_spec(self):
        """Teams, date and at least one lineup player are required; anything
        else may be left blank — missing inputs become NaN features."""
        away, home = self.cb_away.get(), self.cb_home.get()
        if not away or not home or away == home:
            raise ValueError("pick two different teams")
        date = self.e_date.get().strip()
        dt.date.fromisoformat(date)

        def num(widget, label):
            v = str(widget.get()).strip()
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"{label} is not a number: {v!r}")

        start = self.e_start.get().strip()
        if start:
            m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", start, re.I)
            if not m:
                raise ValueError(f"start time not understood: {start!r} "
                                 f"(use 7:10 PM or 19:10)")
            h = int(m.group(1))
            if m.group(3):
                h = h % 12 + (12 if m.group(3).upper() == "PM" else 0)
            if not 0 <= h <= 23:
                raise ValueError(f"start time hour out of range: {start!r}")
            start = f"{h:02d}:{m.group(2)}"
        spec = {"date": date, "away_team": away, "home_team": home,
                "start_et": start or None,
                "venue": self.cb_venue.get().strip(),
                "day_night": self.cb_dn.get(),
                "temp": num(self.sp_temp, "temperature"),
                "wind_speed": num(self.sp_wind, "wind speed"),
                "humidity": num(self.sp_hum, "humidity"),
                "pressure": num(self.e_pres, "pressure"),
                "precip": num(self.e_precip, "precip"),
                "wind_dir": self.cb_wdir.get(), "condition": self.cb_cond.get()}
        ump = self.cb_ump.get().strip()
        spec["hp_ump"] = ump or None
        spec["hp_ump_id"] = self.ump_name_to_id.get(ump)   # None if unknown
        for side, team in (("away", away), ("home", home)):
            w = self.side_widgets[side]
            st = w["starter"].get().strip()
            spec[f"{side}_starter"] = (self._resolve(team, st, "pitchers")
                                       if st else None)
            lineup = []
            for slot, cb in enumerate(w["lineup"], start=1):
                lab = cb.get().strip()
                if lab:
                    lineup.append((self._resolve(team, lab, "batters"), slot))
            if len({p for p, _ in lineup}) != len(lineup):
                raise ValueError(f"duplicate player in {side} lineup")
            spec[f"{side}_lineup"] = lineup
        if not spec["away_lineup"] and not spec["home_lineup"]:
            raise ValueError("fill in at least one lineup player")
        return spec

    @staticmethod
    def _slate_row_text(spec):
        n = len(spec.get("away_lineup", [])) + len(spec.get("home_lineup", []))
        start = spec.get("start_et") or "--:--"
        return (f'{spec["date"]}  {start} ET  {spec["away_team"]} @ '
                f'{spec["home_team"]}  ({n} batters)')

    def _add_to_slate(self):
        try:
            spec = self._collect_spec()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        self.slate.append(spec)
        self.lb_slate.insert("end", self._slate_row_text(spec))
        for side in ("away", "home"):
            w = self.side_widgets[side]
            w["starter"].set("")
            for cb in w["lineup"]:
                cb.set("")
        self.cb_away.set("")
        self.cb_home.set("")
        self._refresh_team_options()
        self.status.set(f"Slate: {len(self.slate)} game(s). Add more or Predict.")

    def _slate_selected(self, _event=None):
        """Clicking a slate game loads it into the form for editing."""
        sel = self.lb_slate.curselection()
        if not sel or not (0 <= sel[0] < len(self.slate)):
            return
        self._loaded_idx = sel[0]
        spec = self.slate[sel[0]]
        self._apply_spec(spec)
        self.status.set(
            f'Loaded {spec["away_team"]} @ {spec["home_team"]} into the form '
            f'— edit, then "Update selected game" to save it back. '
            f'Predict still runs the whole slate.')

    def _update_selected(self):
        """Write the form back over the selected (or last-loaded) slate game."""
        sel = self.lb_slate.curselection()
        idx = sel[0] if sel else self._loaded_idx
        if idx is None or not (0 <= idx < len(self.slate)):
            messagebox.showinfo(
                "No game selected",
                "Click a slate game first — it loads into the form. Edit it, "
                "then Update selected game saves your changes back.")
            return
        try:
            spec = self._collect_spec()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        old = self.slate[idx]
        if old.get("names"):     # keep scraped display names for re-loading
            spec["names"] = old["names"]
        # scraped fields with no form widget — carry them through the edit
        # (DH flags, game identity, lineup provenance, next scheduled off-days)
        for k in ("is_dh", "dh_game2", "game_pk",
                  "away_lineup_src", "home_lineup_src",
                  "away_next_offday", "home_next_offday"):
            if old.get(k) is not None:
                spec[k] = old[k]
        self.slate[idx] = spec
        self.lb_slate.delete(idx)
        self.lb_slate.insert(idx, self._slate_row_text(spec))
        self.lb_slate.selection_clear(0, "end")
        self.lb_slate.selection_set(idx)
        self.lb_slate.see(idx)
        self._loaded_idx = idx
        self.status.set(f'Updated game {idx + 1} of {len(self.slate)}: '
                        f'{spec["away_team"]} @ {spec["home_team"]}.')

    def _remove_from_slate(self):
        sel = self.lb_slate.curselection()
        if sel:
            self.slate.pop(sel[0])
            self.lb_slate.delete(sel[0])
            self._loaded_idx = None

    def _clear_slate(self):
        self.slate.clear()
        self.lb_slate.delete(0, "end")
        self._loaded_idx = None

    def _move_slate(self, delta):
        """Shift the selected slate game up/down — the slate order IS the
        game order of every sheet in the output workbook."""
        sel = self.lb_slate.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if not (0 <= j < len(self.slate)):
            return
        self.slate[i], self.slate[j] = self.slate[j], self.slate[i]
        row = self.lb_slate.get(i)
        self.lb_slate.delete(i)
        self.lb_slate.insert(j, row)
        self.lb_slate.selection_clear(0, "end")
        self.lb_slate.selection_set(j)
        self.lb_slate.see(j)
        self._loaded_idx = None

    def _predict_clicked(self):
        if self.slate:
            specs = list(self.slate)
        else:
            try:
                specs = [self._collect_spec()]
            except Exception as e:
                messagebox.showerror("Input error", str(e))
                return
        self.btn_predict["state"] = "disabled"
        self.status.set(f"Predicting {len(specs)} game(s)...")
        self._pred_state = None
        self._jobs.put(lambda: self._predict_run(specs))
        self.after(200, self._poll_predict)

    def _predict_run(self, specs):
        try:
            from predict import save_excel_slate
            out = self.pred.predict_slate(specs)
            xlsx = save_excel_slate(specs, out)
            self._pred_state = ("ok", (specs, out, xlsx))
        except Exception as e:
            self._pred_state = ("err", str(e))

    def _poll_predict(self):
        if self._pred_state is None:
            self.after(200, self._poll_predict)
            return
        state, payload = self._pred_state
        self.btn_predict["state"] = "normal"
        if state == "ok":
            _specs, _out, xlsx = payload
            self.status.set(f"Saved: {xlsx}")
        else:
            self.status.set("Ready.")
            messagebox.showerror("Prediction failed", payload)

    # ---------------------------------------------------- batch predict

    def _batch_predict(self, paths):
        """Serve several slate files back-to-back on the worker thread.
        Each file gets its own workbook exactly as if it were loaded and
        predicted alone; the slate list in the window is untouched."""
        if self.pred is None:
            messagebox.showinfo(
                "Still loading",
                "Models are still loading — try again once the status "
                "bar says Ready.")
            return
        self.btn_predict["state"] = "disabled"
        self._pred_state = None
        self._batch_msg = f"Batch: starting {len(paths)} slates..."
        self._jobs.put(lambda: self._batch_run(paths))
        self.after(200, self._poll_batch)

    def _batch_run(self, paths):
        from predict import save_excel_slate
        saved, failed = [], []
        for i, path in enumerate(paths, 1):
            try:
                self._batch_msg = (f"Batch {i}/{len(paths)}: {path.name} "
                                   f"— predicting...")
                specs, _ = self._read_slate_specs(path)
                if not specs:
                    failed.append(f"{path.name}: no games")
                    continue
                out = self.pred.predict_slate(specs)
                saved.append(save_excel_slate(specs, out))
            except Exception as e:
                failed.append(f"{path.name}: {e}")
        self._pred_state = ("ok", (saved, failed))

    def _poll_batch(self):
        if self._pred_state is None:
            self.status.set(self._batch_msg)
            self.after(200, self._poll_batch)
            return
        _state, (saved, failed) = self._pred_state
        self.btn_predict["state"] = "normal"
        tail = f", {len(failed)} failed" if failed else ""
        self.status.set(f"Batch done: {len(saved)} workbook(s) saved"
                        f"{tail}. Last: {saved[-1] if saved else '—'}")
        if failed:
            messagebox.showwarning("Batch finished with errors",
                                   "\n".join(failed))


if __name__ == "__main__":
    App().mainloop()
