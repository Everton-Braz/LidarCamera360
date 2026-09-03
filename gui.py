# -*- coding: utf-8 -*-
"""Lidar-Camera calibrator - user interface."""

import json
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

C_BG = "#0A0A0D"
C_PANEL = "#14141A"
C_PANEL2 = "#1A1A22"
C_BTN = "#1F1F26"
C_HOVER = "#2A2A34"
C_BORDER = "#2A2A34"
C_DIM = "#55555F"
C_HINT = "#888892"
C_SILVER = "#B8B8C0"
C_LIGHT = "#E8E8F0"
C_TEXT = "#E0E0E8"
C_GREEN = "#5AA86A"
C_RED = "#C05050"
C_AMBER = "#C8A34A"

FONT = "Segoe UI"
MAX_SETS = 5
DEFAULT_ON = 2
TIP_MS = 10000

HELP = {
    "up": "UP - how far the camera sits ABOVE the lidar, in metres.\n"
          "Negative if the camera is below the lidar.",
    "back": "BACK - how far the camera sits BEHIND the lidar, in metres.\n"
            "Behind means against the direction the camera looks.\n"
            "Negative if the camera is in front of the lidar.",
    "right": "RIGHT - how far the camera sits to the RIGHT of the lidar, in metres.\n"
             "Right as seen looking along the camera axis.\n"
             "Negative if the camera is to the left.",
    "tol": "TOLERANCE - half-width of the accepted box, in metres.\n"
           "The guesses only reject grossly wrong minima, so rough numbers are\n"
           "fine; the true value must fall inside guess +/- tolerance.",
}

ctk.set_appearance_mode("dark")


def make_logo(parent, size=34):
    cv = tk.Canvas(parent, width=size, height=size, bg=C_BG, highlightthickness=0)
    cv.create_oval(2, 2, size - 2, size - 2, outline=C_SILVER, width=2)
    cv.create_text(size / 2, size / 2 + 1, text="LC", fill=C_LIGHT,
                   font=(FONT, int(size * 0.36), "bold"))
    return cv


def make_icon(size=64):
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageTk
    except ImportError:
        return None
    img = Image.new("RGBA", (size, size), (10, 10, 13, 255))
    d = ImageDraw.Draw(img)
    m = int(size * 0.08)
    d.ellipse([m, m, size - m, size - m], outline=(184, 184, 192, 255),
              width=max(2, size // 18))
    try:
        f = ImageFont.truetype("segoeuib.ttf", int(size * 0.36))
        box = d.textbbox((0, 0), "LC", font=f)
        d.text(((size - box[2] + box[0]) / 2 - box[0],
                (size - box[3] + box[1]) / 2 - box[1]), "LC", font=f,
               fill=(232, 232, 240, 255))
    except OSError:
        pass
    return ImageTk.PhotoImage(img)


class Tip(ctk.CTkToplevel):
    """Popup that closes itself after TIP_MS."""

    _open = None

    def __init__(self, master, text, x, y):
        super().__init__(master)
        if Tip._open is not None:
            try:
                Tip._open.destroy()
            except tk.TclError:
                pass
        Tip._open = self
        self.overrideredirect(True)
        self.configure(fg_color=C_PANEL2)
        self.attributes("-topmost", True)
        frm = ctk.CTkFrame(self, fg_color=C_PANEL2, corner_radius=6,
                           border_width=1, border_color=C_BORDER)
        frm.pack(fill="both", expand=True)
        ctk.CTkLabel(frm, text=text, justify="left", font=(FONT, 12),
                     text_color=C_TEXT, wraplength=400).pack(padx=12, pady=(9, 6))
        ctk.CTkLabel(frm, text="closes in 10 s  ·  click to dismiss",
                     font=(FONT, 10), text_color=C_DIM).pack(padx=12, pady=(0, 8))
        self.geometry("+%d+%d" % (x, y))
        for w in (self, frm, *frm.winfo_children()):
            w.bind("<Button-1>", lambda e: self.close())
        self._job = self.after(TIP_MS, self.close)

    def close(self):
        try:
            self.after_cancel(self._job)
        except (tk.TclError, ValueError):
            pass
        if Tip._open is self:
            Tip._open = None
        self.destroy()


class SetRow(ctk.CTkFrame):
    """Checkbox, one bag and the matching clip."""

    def __init__(self, master, index, on_toggle):
        super().__init__(master, fg_color=C_PANEL2, corner_radius=6,
                         border_width=1, border_color=C_BORDER)
        self.index = index
        self.bag = ""
        self.video = ""
        self.on_toggle = on_toggle
        self.var = ctk.BooleanVar(value=index <= DEFAULT_ON)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(4, weight=1)

        self.chk = ctk.CTkCheckBox(self, text="", width=24, variable=self.var,
                                   fg_color=C_GREEN, hover_color=C_GREEN,
                                   border_color=C_BORDER, checkbox_width=18,
                                   checkbox_height=18, command=self._toggled)
        self.chk.grid(row=0, column=0, padx=(12, 2), pady=8)
        self.num = ctk.CTkLabel(self, text=str(index), width=18, text_color=C_HINT,
                                font=(FONT, 12, "bold"))
        self.num.grid(row=0, column=1, padx=(0, 6))
        self.b_bag = self._btn("lidar .bag ...", self.pick_bag)
        self.b_bag.grid(row=0, column=2, sticky="ew", padx=3, pady=8)
        ctk.CTkLabel(self, text="+", text_color=C_DIM,
                     font=(FONT, 13)).grid(row=0, column=3, padx=2)
        self.b_vid = self._btn("360 .mp4, no stabilisation ...", self.pick_video)
        self.b_vid.grid(row=0, column=4, sticky="ew", padx=3, pady=8)
        self.status = ctk.CTkLabel(self, text="", width=110, text_color=C_HINT,
                                   font=(FONT, 11))
        self.status.grid(row=0, column=5, padx=(4, 12))
        self._apply_state()

    def _btn(self, text, cmd):
        return ctk.CTkButton(self, text=text, height=28, anchor="w", font=(FONT, 12),
                             fg_color=C_BTN, hover_color=C_HOVER, text_color=C_SILVER,
                             corner_radius=4, border_width=1, border_color=C_BORDER,
                             command=cmd)

    def _toggled(self):
        self._apply_state()
        self.on_toggle()

    def _apply_state(self):
        on = self.var.get()
        state = "normal" if on else "disabled"
        self.b_bag.configure(state=state)
        self.b_vid.configure(state=state)
        self.num.configure(text_color=C_SILVER if on else C_DIM)
        self.configure(border_color=C_BORDER if on else "#17171D")
        if on:
            self.refresh()
        else:
            self.set_status("off", C_DIM)

    @staticmethod
    def _short(p, n=34):
        b = os.path.basename(p)
        return b if len(b) <= n else b[:n - 9] + "..." + b[-6:]

    def pick_bag(self):
        p = filedialog.askopenfilename(title="Static capture (.bag)",
                                       filetypes=[("ROS bag", "*.bag"), ("all", "*.*")])
        if p:
            self.bag = p
            self.b_bag.configure(text="  " + self._short(p), text_color=C_LIGHT)
            self.refresh()

    def pick_video(self):
        p = filedialog.askopenfilename(title="360 mp4 exported without stabilisation",
                                       filetypes=[("MP4", "*.mp4"), ("all", "*.*")])
        if p:
            self.video = p
            self.b_vid.configure(text="  " + self._short(p), text_color=C_LIGHT)
            self.refresh()

    def refresh(self):
        if self.var.get():
            self.set_status("ready" if self.ok() else "", C_GREEN if self.ok() else C_HINT)

    def set_status(self, text, color=C_HINT):
        self.status.configure(text=text, text_color=color)

    def enabled(self):
        return bool(self.var.get())

    def ok(self):
        return bool(self.bag and self.video)


class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=C_BG)
        self.title("Lidar-Camera calibrator")
        self.geometry("1120x900")
        self.minsize(1040, 800)
        self._icon = make_icon()
        if self._icon is not None:
            try:
                self.iconphoto(True, self._icon)
            except tk.TclError:
                pass

        self.q = queue.Queue()
        self.worker = None
        self.stop_flag = False
        self.solver = None
        self.result_json = None

        self._header()
        self._sets()
        self._prior()
        self._actions()
        self._output()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(80, self.pump)

    # ------------------------------------------------------------ layout --
    def _line(self):
        ctk.CTkFrame(self, height=1, fg_color=C_BORDER,
                     corner_radius=0).pack(fill="x", padx=22, pady=6)

    def _header(self):
        h = ctk.CTkFrame(self, fg_color=C_BG)
        h.pack(fill="x", padx=22, pady=(18, 6))
        make_logo(h).pack(side="left", padx=(0, 12))
        t = ctk.CTkFrame(h, fg_color=C_BG)
        t.pack(side="left")
        ctk.CTkLabel(t, text="Raven LiDAR & Insta360 Calibrator", font=(FONT, 21, "bold"),
                     text_color=C_LIGHT).pack(anchor="w")
        ctk.CTkLabel(t, text="3DMakerPro Raven (Vanjee 722z) to Insta360 X4 360° extrinsics",
                     font=(FONT, 12), text_color=C_HINT).pack(anchor="w")

        right = ctk.CTkFrame(h, fg_color=C_BG)
        right.pack(side="right")
        ctk.CTkLabel(right, text="frame, %", font=(FONT, 11),
                     text_color=C_DIM).pack(side="left", padx=(0, 4))
        self.e_frame = ctk.CTkEntry(right, width=54, height=28, font=(FONT, 12),
                                    justify="center", fg_color=C_BTN,
                                    border_color=C_BORDER, border_width=1,
                                    text_color=C_LIGHT, corner_radius=4)
        self.e_frame.insert(0, "50")
        self.e_frame.pack(side="left", padx=(0, 12))
        self.quality = ctk.StringVar(value="Normal")
        ctk.CTkLabel(right, text="quality", font=(FONT, 11),
                     text_color=C_DIM).pack(side="left", padx=(0, 4))
        ctk.CTkOptionMenu(right, values=["Fast", "Normal", "High"], variable=self.quality,
                          width=96, height=28, font=(FONT, 12), fg_color=C_BTN,
                          button_color=C_BTN, button_hover_color=C_HOVER,
                          text_color=C_SILVER, dropdown_fg_color=C_PANEL2,
                          dropdown_hover_color=C_HOVER, dropdown_text_color=C_SILVER,
                          corner_radius=4).pack(side="left")

    def _sets(self):
        self._line()
        head = ctk.CTkFrame(self, fg_color=C_BG)
        head.pack(fill="x", padx=22, pady=(4, 2))
        ctk.CTkLabel(head, text="DATA SETS", font=(FONT, 11, "bold"),
                     text_color=C_SILVER).pack(side="left")
        ctk.CTkLabel(head, text="  static capture · tick a box to add another set",
                     font=(FONT, 11), text_color=C_DIM).pack(side="left")
        self.lbl_count = ctk.CTkLabel(head, text="", font=(FONT, 11), text_color=C_HINT)
        self.lbl_count.pack(side="right")

        box = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=6)
        box.pack(fill="x", padx=22, pady=(2, 8))
        self.rows = []
        for i in range(1, MAX_SETS + 1):
            r = SetRow(box, i, self._count)
            r.pack(fill="x", padx=6, pady=3)
            self.rows.append(r)
        self._count()

    def _count(self):
        n = sum(1 for r in self.rows if r.enabled())
        self.lbl_count.configure(text="%d of %d enabled" % (n, MAX_SETS))

    def _prior(self):
        self._line()
        ctk.CTkLabel(self, text="INITIAL LINEAR GUESS  ·  camera relative to lidar",
                     font=(FONT, 11, "bold"),
                     text_color=C_SILVER).pack(anchor="w", padx=22)
        ctk.CTkLabel(self, text="only used to reject grossly wrong minima - rough numbers are fine",
                     font=(FONT, 11), text_color=C_DIM).pack(anchor="w", padx=22, pady=(0, 4))
        p = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=6)
        p.pack(fill="x", padx=22, pady=(0, 8))
        self.e_up = self._num(p, "up  (m)", "0.18", 0, "above the lidar", "up")
        self.e_back = self._num(p, "back  (m)", "0.07", 1, "behind the lidar", "back")
        self.e_right = self._num(p, "right  (m)", "0.00", 2, "right of the lidar", "right")
        self.e_tol = self._num(p, "tolerance  (m)", "0.12", 3, "half-width of the box", "tol")

    def _num(self, parent, label, default, col, hint, key):
        f = ctk.CTkFrame(parent, fg_color=C_PANEL)
        f.grid(row=0, column=col, padx=14, pady=10, sticky="w")
        top = ctk.CTkFrame(f, fg_color=C_PANEL)
        top.pack(anchor="w")
        ctk.CTkLabel(top, text=label, font=(FONT, 11), text_color=C_HINT).pack(side="left")
        q = ctk.CTkButton(top, text="?", width=18, height=18, corner_radius=9,
                          font=(FONT, 10, "bold"), fg_color=C_BTN, hover_color=C_HOVER,
                          text_color=C_HINT, border_width=1, border_color=C_BORDER)
        q.configure(command=lambda w=q, k=key: self._help(w, k))
        q.pack(side="left", padx=(6, 0))
        e = ctk.CTkEntry(f, width=100, height=28, font=(FONT, 13), justify="center",
                         fg_color=C_BTN, border_color=C_BORDER, border_width=1,
                         text_color=C_LIGHT, corner_radius=4)
        e.insert(0, default)
        e.pack(anchor="w", pady=(2, 1))
        ctk.CTkLabel(f, text=hint, font=(FONT, 10), text_color=C_DIM).pack(anchor="w")
        return e

    def _help(self, widget, key):
        Tip(self, HELP[key], widget.winfo_rootx() + 24, widget.winfo_rooty() + 20)

    def _actions(self):
        a = ctk.CTkFrame(self, fg_color=C_BG)
        a.pack(fill="x", padx=22, pady=(2, 6))
        self.b_run = ctk.CTkButton(a, text="CALIBRATE", width=170, height=38,
                                   font=(FONT, 13, "bold"), fg_color=C_GREEN,
                                   hover_color="#4C9159", text_color="#0A0A0D",
                                   corner_radius=5, command=self.start)
        self.b_run.pack(side="left")
        self.b_save = ctk.CTkButton(a, text="SAVE JSON", width=140, height=38,
                                    font=(FONT, 13, "bold"), fg_color=C_BTN,
                                    hover_color=C_HOVER, text_color=C_DIM, corner_radius=5,
                                    border_width=1, border_color=C_BORDER,
                                    state="disabled", command=self.save)
        self.b_save.pack(side="left", padx=10)
        self.b_tags = ctk.CTkButton(a, text="APRILTAG TARGETS", width=160, height=38,
                                    font=(FONT, 12, "bold"), fg_color=C_BTN,
                                    hover_color=C_HOVER, text_color=C_SILVER, corner_radius=5,
                                    border_width=1, border_color=C_BORDER,
                                    command=self.open_apriltag_targets)
        self.b_tags.pack(side="left", padx=(0, 10))
        self.prog = ctk.CTkProgressBar(a, height=8, corner_radius=3, fg_color=C_PANEL2,
                                       progress_color=C_GREEN)
        self.prog.set(0)
        self.prog.pack(side="left", fill="x", expand=True, padx=(14, 8))
        self.lbl_state = ctk.CTkLabel(a, text="idle", font=(FONT, 11),
                                      text_color=C_HINT, width=200)
        self.lbl_state.pack(side="right")

    def open_apriltag_targets(self):
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apriltags_to_print")
        if not os.path.exists(target_dir):
            import generate_apriltags
            os.system(f'python "{os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_apriltags.py")}" --count 12 --size 150 --out "{target_dir}"')
        try:
            os.startfile(target_dir)
        except Exception:
            messagebox.showinfo("AprilTag Targets", f"Printable targets generated at:\n{target_dir}")

    def _output(self):
        self._line()
        tabs = ctk.CTkTabview(self, fg_color=C_PANEL, segmented_button_fg_color=C_PANEL,
                              segmented_button_selected_color=C_HOVER,
                              segmented_button_selected_hover_color=C_HOVER,
                              segmented_button_unselected_color=C_PANEL,
                              segmented_button_unselected_hover_color=C_BTN,
                              text_color=C_SILVER, corner_radius=6)
        tabs.pack(fill="both", expand=True, padx=22, pady=(2, 18))
        tabs.add("RESULT")
        tabs.add("LOG")
        self.tabs = tabs
        self.txt_res = self._mono(tabs.tab("RESULT"))
        self.txt_log = self._mono(tabs.tab("LOG"))
        self._res("No calibration yet.\n\n"
                  "1.  pick a .bag and the matching .mp4 for every enabled set\n"
                  "2.  the clip must be exported WITHOUT stabilisation or horizon lock\n"
                  "3.  adjust the linear guesses if your rig differs\n"
                  "4.  press CALIBRATE\n")

    def _mono(self, parent):
        t = tk.Text(parent, bg=C_PANEL, fg=C_TEXT, insertbackground=C_SILVER,
                    font=("Consolas", 11), relief="flat", wrap="none",
                    padx=12, pady=10, borderwidth=0)
        sy = tk.Scrollbar(parent, command=t.yview, bg=C_PANEL, troughcolor=C_PANEL,
                          borderwidth=0, highlightthickness=0)
        t.configure(yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        for tag, col in (("ok", C_GREEN), ("bad", C_RED), ("warn", C_AMBER),
                         ("dim", C_HINT), ("hi", C_LIGHT)):
            t.tag_config(tag, foreground=col)
        return t

    # ---------------------------------------------------------- plumbing --
    def _res(self, text, tag=None):
        self.txt_res.delete("1.0", "end")
        self.txt_res.insert("end", text, tag or ())

    def log(self, msg):
        self.q.put(("log", str(msg)))

    def pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.txt_log.insert("end", payload + "\n")
                    self.txt_log.see("end")
                elif kind == "progress":
                    self.prog.set(max(0.0, min(1.0, payload)))
                elif kind == "state":
                    self.lbl_state.configure(text=payload)
                elif kind == "done":
                    self.finish(payload)
                elif kind == "fail":
                    self.fail(payload)
        except queue.Empty:
            pass
        self.after(80, self.pump)

    # -------------------------------------------------------------- run --
    def start(self):
        if self.worker and self.worker.is_alive():
            self.stop_flag = True
            self.lbl_state.configure(text="stopping ...")
            return
        rows = [r for r in self.rows if r.enabled()]
        if not rows:
            messagebox.showwarning("Calibrator", "No data set is enabled.")
            return
        for r in rows:
            if not r.ok():
                messagebox.showwarning("Calibrator", "Set %d is incomplete." % r.index)
                return
        try:
            vals = [float(e.get().replace(",", ".")) for e in
                    (self.e_up, self.e_back, self.e_right, self.e_tol)]
            fpos = float(self.e_frame.get().replace(",", ".")) / 100.0
        except ValueError:
            messagebox.showwarning("Calibrator",
                                   "The guesses and the frame position must be numbers.")
            return
        if vals[3] <= 0:
            messagebox.showwarning("Calibrator", "Tolerance must be greater than zero.")
            return
        if not 0.0 < fpos < 1.0:
            messagebox.showwarning("Calibrator", "Frame position must be between 1 and 99 %.")
            return

        self.stop_flag = False
        self.result_json = None
        self.txt_log.delete("1.0", "end")
        self._res("Working ...\n")
        self.b_save.configure(state="disabled", text_color=C_DIM)
        self.b_run.configure(text="STOP", fg_color=C_RED, hover_color="#A34444",
                             text_color=C_LIGHT)
        for r in rows:
            r.set_status("queued", C_HINT)
        self.worker = threading.Thread(
            target=self._work, args=([(r.bag, r.video) for r in rows], vals, fpos),
            daemon=True)
        self.worker.start()

    def _work(self, pairs, vals, fpos):
        try:
            from core import Prior
            from solver import Solver
            self.q.put(("state", "loading data ..."))
            s = Solver(pairs, Prior(up=vals[0], back=vals[1], right=vals[2], tol=vals[3]),
                       quality=self.quality.get(), log=self.log,
                       progress=lambda f: self.q.put(("progress", f)),
                       should_stop=lambda: self.stop_flag, frame_pos=fpos)
            self.solver = s
            self.q.put(("done", s.run()))
        except Exception as e:
            if type(e).__name__ == "Cancelled":
                self.q.put(("fail", "stopped by the user"))
            else:
                self.log(traceback.format_exc())
                self.q.put(("fail", str(e)))

    # ---------------------------------------------------------- results --
    def finish(self, payload):
        import numpy as np
        from scipy.spatial.transform import Rotation as Rot

        ok, bad = payload
        s = self.solver
        self.b_run.configure(text="CALIBRATE", fg_color=C_GREEN, hover_color="#4C9159",
                             text_color="#0A0A0D")
        self.prog.set(1.0)
        contradiction = s.stats.get("contradiction", False)
        self.lbl_state.configure(text="SETS CONTRADICT" if contradiction
                                 else "%d accepted, %d rejected" % (len(ok), len(bad)))

        for r in self.rows:
            for d in s.sets:
                if d.bag == r.bag:
                    if getattr(d, "pair_suspect", False):
                        r.set_status("pairing?", C_AMBER)
                    elif d.rejected:
                        r.set_status("rejected", C_RED)
                    else:
                        r.set_status("ok  %.3f" % d.solo_score, C_GREEN)

        t = self.txt_res
        t.delete("1.0", "end")
        suspect = [d for d in s.sets if getattr(d, "pair_suspect", False)]
        if suspect:
            t.insert("end", "WARNING: bag and mp4 may not be from the same capture\n", "warn")
            for d in suspect:
                t.insert("end", "   %s - the cloud does not lock onto this video "
                                "(peak %.3f, margin %.2f sigma)\n"
                         % (d.name, getattr(d, "pair_score", 0),
                            getattr(d, "pair_margin", 0)), "warn")
            t.insert("end", "\n")
        if contradiction:
            t.insert("end", "THE SETS COULD NOT BE RECONCILED\n", "bad")
            t.insert("end", "No answer explains all of them. Per-set results follow.\n\n", "bad")

        t.insert("end", "PER SET\n", "hi")
        t.insert("end", "-" * 122 + "\n", "dim")
        t.insert("end", "%-26s %-9s %-24s %-24s %-19s %-7s\n" %
                 ("set", "status", "yaw/pitch/roll  deg", "C in lidar frame  m",
                  "up/back/right  cm", "score"), "dim")
        for name, status, rot, C, rig, score, note in s.summary_rows():
            t.insert("end", "%-26s " % name[:26])
            t.insert("end", "%-9s " % status, "ok" if status == "OK" else "bad")
            t.insert("end", "%-24s %-24s %-19s %-7s\n" % (rot, C, rig, score))
            if note:
                t.insert("end", "   %s\n" % note,
                         "warn" if note.startswith("PAIRING")
                         else ("dim" if status == "OK" else "bad"))

        cross = s.stats.get("cross_check")
        if cross:
            t.insert("end", "\nCROSS-CHECK  (each answer scored on every set)\n", "hi")
            t.insert("end", "-" * 122 + "\n", "dim")
            idx, M = cross["idx"], cross["matrix"]
            t.insert("end", "        " + "".join("  set%-6d" % i for i in idx) + "\n", "dim")
            for a, i in enumerate(idx):
                t.insert("end", "set%-5d " % i +
                         "".join("  %8.4f" % M[a][b] for b in range(len(idx))) + "\n")

        if s.R is None:
            return
        e = Rot.from_matrix(s.R).as_euler("ZYX", degrees=True)
        rig = s.stats["rig"]
        t.insert("end", "\n%s   (%d set(s))\n"
                 % ("BEST SINGLE SET" if contradiction else "CONSENSUS",
                    s.stats["n_ok"]), "hi")
        t.insert("end", "-" * 122 + "\n", "dim")
        t.insert("end", "   camera above lidar   %+7.1f cm\n" % (100 * rig["up"]))
        t.insert("end", "   camera behind        %+7.1f cm\n" % (100 * rig["back"]))
        t.insert("end", "   camera to the right  %+7.1f cm\n" % (100 * rig["right"]))
        t.insert("end", "   euler ZYX            yaw %.3f   pitch %.3f   roll %.3f\n"
                 % (e[0], e[1], e[2]))
        t.insert("end", "   C in lidar frame     %s\n" % np.round(s.C, 5).tolist())
        t.insert("end", "   alignment score      %.4f\n" % s.stats["score"])
        t.insert("end", "   heading margin       %.2f sigma\n" % s.stats["joint_margin"])
        t.insert("end", "   agreement            %s\n" % s.stats["uncertainty"])
        if "lsq_residual_deg" in s.stats:
            t.insert("end", "   least-squares resid  %.2f deg / %.0f mm\n"
                     % (s.stats["lsq_residual_deg"], 1000 * s.stats["lsq_residual_m"]))
        t.insert("end", "   zenith cross-check   %.2f deg\n" % s.stats["zenith_resid_deg"])
        t.insert("end", "\n   T_base_camera_lidar\n", "hi")
        for row in s.T:
            t.insert("end", "      [%s]\n" % "  ".join("%12.8f" % v for v in row))
        if s.stats["n_ok"] == 1 and not contradiction:
            t.insert("end", "\n   only one set was accepted, so nothing cross-checked it.\n"
                            "   Add a second good set before trusting these numbers.\n", "warn")

        self.result_json = s.json()
        self.b_save.configure(state="normal", text_color=C_LIGHT)
        self.tabs.set("RESULT")

    def fail(self, msg):
        self.b_run.configure(text="CALIBRATE", fg_color=C_GREEN, hover_color="#4C9159",
                             text_color="#0A0A0D")
        self.lbl_state.configure(text="failed")
        self.prog.set(0)
        self._res("FAILED\n\n%s\n\nSee the LOG tab for details.\n" % msg)
        self.txt_res.tag_add("bad", "1.0", "2.0")

    def save(self):
        if not self.result_json:
            return
        p = filedialog.asksaveasfilename(title="Save extrinsics", defaultextension=".json",
                                         initialfile="extrinsics_determined.json",
                                         filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.result_json, f, indent=2, ensure_ascii=False)
            self.lbl_state.configure(text="saved")
            messagebox.showinfo("Calibrator", "Saved:\n%s" % p)
        except OSError as e:
            messagebox.showerror("Calibrator", "Could not save:\n%s" % e)

    def on_close(self):
        self.stop_flag = True
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
