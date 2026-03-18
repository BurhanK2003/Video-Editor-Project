from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .batch_runner import run_batch_auto_edit
from .models import AutoEditRequest
from .orchestrator import run_auto_edit

RENDER_SPEED_PRESETS = {
    "Fast": "veryfast",
    "Balanced": "medium",
    "Quality": "slow",
}

TRANSITION_STYLE_MAP = {
    "None": "none",
    "Professional Weighted": "pro_weighted",
}

CAPTION_STYLE_MAP = {
    "Bold Stroke": "bold_stroke",
    "Yellow Active": "yellow_active",
    "Gradient Fill": "gradient_fill",
}

AUDIO_FILE_TYPES = "*.wav *.mp3 *.m4a *.aac *.flac *.ogg"


class AutoEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Local Auto Video Editor")
        self.root.geometry("980x700")
        self.root.minsize(900, 620)

        self.voiceover_var = tk.StringVar()
        self.clips_var = tk.StringVar()
        self.music_var = tk.StringVar()
        self.stock_keywords_var = tk.StringVar()
        self.output_var = tk.StringVar(
            value=str((Path.cwd() / "output" / "edited_video.mp4").resolve())
        )
        self.width_var = tk.IntVar(value=1080)
        self.height_var = tk.IntVar(value=1920)
        self.fps_var = tk.IntVar(value=24)
        self.render_speed_var = tk.StringVar(value="Fast")
        self.allow_stock_fetch_var = tk.BooleanVar(value=True)
        self.transition_style_var = tk.StringVar(value="Professional Weighted")
        self.transition_duration_var = tk.DoubleVar(value=0.22)
        self.caption_style_var = tk.StringVar(value="Bold Stroke")
        self.whisper_model_var = tk.StringVar(value="base")
        self.caption_position_ratio_var = tk.DoubleVar(value=0.54)
        self.caption_max_lines_var = tk.IntVar(value=3)
        self.caption_font_scale_var = tk.DoubleVar(value=1.00)
        self.caption_pop_scale_var = tk.DoubleVar(value=1.00)
        self.adaptive_safe_zones_var = tk.BooleanVar(value=True)
        self.karaoke_highlight_var = tk.BooleanVar(value=True)
        self.enable_motion_overlays_var = tk.BooleanVar(value=False)
        self.stat_badge_text_var = tk.StringVar()
        self.cta_text_var = tk.StringVar()
        self.logo_path_var = tk.StringVar()
        self.enable_progress_bar_var = tk.BooleanVar(value=True)
        self.batch_voiceovers_var = tk.StringVar()
        self.batch_manifest_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str((Path.cwd() / "output" / "batch").resolve()))
        # Script-to-video
        self.script_text_var = tk.StringVar()
        self.script_voice_var = tk.StringVar(value="en-US-AriaNeural")

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_canvas_window_width(event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        frame.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_canvas_window_width)

        # Scroll using mouse wheel while pointer is over the editor content.
        def _on_mousewheel(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        split = ttk.Panedwindow(frame, orient="vertical")
        split.pack(fill="both", expand=True)

        controls_pane = ttk.Frame(split)
        logs_pane = ttk.Frame(split)
        split.add(controls_pane, weight=3)
        split.add(logs_pane, weight=2)

        controls_canvas = tk.Canvas(controls_pane, highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(
            controls_pane, orient="vertical", command=controls_canvas.yview
        )
        controls_frame = ttk.Frame(controls_canvas)

        controls_window = controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scrollbar.pack(side="right", fill="y")

        controls_frame.bind(
            "<Configure>",
            lambda event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )
        self._bind_mousewheel_scrolling(controls_canvas, controls_frame)

        self._path_row(
            controls_frame,
            label="Voiceover",
            text_var=self.voiceover_var,
            browse_command=self._pick_voiceover,
        )

        # Script-to-video section (collapsed by default via LabelFrame).
        script_frame = ttk.LabelFrame(controls_frame, text="Script-to-Video  (leave Voiceover empty to use TTS)", padding=6)
        script_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(script_frame, text="Script").grid(row=0, column=0, padx=6, sticky="nw")
        self.script_text_widget = tk.Text(script_frame, height=3, wrap="word")
        self.script_text_widget.grid(row=0, column=1, columnspan=3, padx=6, sticky="ew")
        script_frame.columnconfigure(1, weight=1)
        ttk.Label(script_frame, text="Voice").grid(row=1, column=0, padx=6, pady=(4, 0), sticky="w")
        ttk.Entry(script_frame, textvariable=self.script_voice_var, width=28).grid(
            row=1, column=1, padx=6, pady=(4, 0), sticky="w"
        )
        ttk.Label(script_frame, text="e.g. en-US-GuyNeural", foreground="grey").grid(
            row=1, column=2, padx=4, pady=(4, 0), sticky="w"
        )

        self._path_row(
            controls_frame,
            label="Clips Folder (optional)",
            text_var=self.clips_var,
            browse_command=self._pick_clips_folder,
        )
        self._entry_row(
            controls_frame,
            label="Stock Search",
            text_var=self.stock_keywords_var,
        )
        self._path_row(
            controls_frame,
            label="Music (File/Folder)",
            text_var=self.music_var,
            browse_command=self._pick_music_source,
        )
        self._path_row(
            controls_frame,
            label="Output File",
            text_var=self.output_var,
            browse_command=self._pick_output,
        )

        batch = ttk.LabelFrame(controls_frame, text="Batch Mode", padding=8)
        batch.pack(fill="x", pady=(0, 8))

        self._path_row(
            batch,
            label="Voiceovers Folder",
            text_var=self.batch_voiceovers_var,
            browse_command=self._pick_batch_voiceovers_folder,
        )
        self._path_row(
            batch,
            label="Manifest CSV (opt)",
            text_var=self.batch_manifest_var,
            browse_command=self._pick_batch_manifest,
        )
        self._path_row(
            batch,
            label="Batch Output",
            text_var=self.batch_output_var,
            browse_command=self._pick_batch_output_folder,
        )
        ttk.Label(
            batch,
            text="CSV columns: voiceover/filename, title, keywords, caption_style, transition_style",
            foreground="grey",
        ).pack(anchor="w", padx=4, pady=(2, 4))

        settings = ttk.LabelFrame(controls_frame, text="Render Settings", padding=8)
        settings.pack(fill="x", pady=(10, 8))

        ttk.Label(settings, text="Width").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Spinbox(settings, from_=320, to=7680, textvariable=self.width_var, width=10).grid(
            row=0, column=1, padx=8, pady=6, sticky="w"
        )

        ttk.Label(settings, text="Height").grid(row=0, column=2, padx=8, pady=6, sticky="w")
        ttk.Spinbox(settings, from_=240, to=4320, textvariable=self.height_var, width=10).grid(
            row=0, column=3, padx=8, pady=6, sticky="w"
        )

        ttk.Label(settings, text="FPS").grid(row=0, column=4, padx=8, pady=6, sticky="w")
        ttk.Spinbox(settings, from_=12, to=120, textvariable=self.fps_var, width=10).grid(
            row=0, column=5, padx=8, pady=6, sticky="w"
        )

        ttk.Label(settings, text="Speed").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.render_speed_var,
            values=list(RENDER_SPEED_PRESETS.keys()),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, padx=8, pady=6, sticky="w")

        ttk.Checkbutton(
            settings,
            text="Fetch stock clips only for weak local matches, then match local + stock",
            variable=self.allow_stock_fetch_var,
        ).grid(row=2, column=0, columnspan=6, padx=8, pady=(2, 6), sticky="w")

        ttk.Label(settings, text="Transition").grid(row=3, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.transition_style_var,
            values=list(TRANSITION_STYLE_MAP.keys()),
            state="readonly",
            width=22,
        ).grid(row=3, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(settings, text="Engine").grid(row=3, column=2, padx=8, pady=6, sticky="w")
        ttk.Label(settings, text="35/25/20/10/10 weighted cut mix").grid(
            row=3, column=3, columnspan=3, padx=8, pady=6, sticky="w"
        )

        ttk.Label(
            settings,
            text="zoom punch 6f | smash 1f | whip 4f | glitch 3f | fade 2-4f",
            foreground="grey",
        ).grid(row=4, column=0, columnspan=6, padx=8, pady=(0, 6), sticky="w")

        ttk.Label(settings, text="Captions").grid(row=5, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.caption_style_var,
            values=list(CAPTION_STYLE_MAP.keys()),
            state="readonly",
            width=12,
        ).grid(row=5, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(settings, text="Whisper").grid(row=5, column=2, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.whisper_model_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
            width=10,
        ).grid(row=5, column=3, padx=8, pady=6, sticky="w")

        subtitle_style = ttk.LabelFrame(controls_frame, text="Subtitle Styling Editor", padding=8)
        subtitle_style.pack(fill="x", pady=(0, 8))

        ttk.Label(subtitle_style, text="Vertical Pos").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Spinbox(
            subtitle_style,
            from_=0.40,
            to=0.75,
            increment=0.01,
            textvariable=self.caption_position_ratio_var,
            width=8,
            format="%.2f",
        ).grid(row=0, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(subtitle_style, text="Max Lines").grid(row=0, column=2, padx=8, pady=6, sticky="w")
        ttk.Spinbox(
            subtitle_style,
            from_=1,
            to=5,
            increment=1,
            textvariable=self.caption_max_lines_var,
            width=8,
            format="%.0f",
        ).grid(row=0, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(subtitle_style, text="Font Scale").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Spinbox(
            subtitle_style,
            from_=0.70,
            to=1.60,
            increment=0.05,
            textvariable=self.caption_font_scale_var,
            width=8,
            format="%.2f",
        ).grid(row=1, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(subtitle_style, text="Pop Scale").grid(row=1, column=2, padx=8, pady=6, sticky="w")
        ttk.Spinbox(
            subtitle_style,
            from_=0.60,
            to=1.80,
            increment=0.05,
            textvariable=self.caption_pop_scale_var,
            width=8,
            format="%.2f",
        ).grid(row=1, column=3, padx=8, pady=6, sticky="w")

        ttk.Checkbutton(
            subtitle_style,
            text="Adaptive safe zones",
            variable=self.adaptive_safe_zones_var,
        ).grid(row=2, column=0, columnspan=2, padx=8, pady=(2, 6), sticky="w")

        ttk.Checkbutton(
            subtitle_style,
            text="Per-word karaoke highlight",
            variable=self.karaoke_highlight_var,
        ).grid(row=2, column=2, columnspan=2, padx=8, pady=(2, 6), sticky="w")

        overlays = ttk.LabelFrame(controls_frame, text="Motion Graphics Overlays", padding=8)
        overlays.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(
            overlays,
            text="Enable motion graphics overlays",
            variable=self.enable_motion_overlays_var,
        ).grid(row=0, column=0, columnspan=4, padx=8, pady=(2, 6), sticky="w")

        ttk.Label(overlays, text="Stat Badge").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(overlays, textvariable=self.stat_badge_text_var, width=26).grid(
            row=1, column=1, padx=8, pady=6, sticky="w"
        )

        ttk.Label(overlays, text="CTA").grid(row=1, column=2, padx=8, pady=6, sticky="w")
        ttk.Entry(overlays, textvariable=self.cta_text_var, width=18).grid(
            row=1, column=3, padx=8, pady=6, sticky="w"
        )

        ttk.Label(overlays, text="Logo PNG").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(overlays, textvariable=self.logo_path_var).grid(
            row=2, column=1, columnspan=2, padx=8, pady=6, sticky="ew"
        )
        ttk.Button(overlays, text="Browse", command=self._pick_logo).grid(
            row=2, column=3, padx=8, pady=6, sticky="w"
        )
        overlays.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            overlays,
            text="Top progress bar",
            variable=self.enable_progress_bar_var,
        ).grid(row=3, column=0, columnspan=4, padx=8, pady=(2, 6), sticky="w")

        self.run_button = ttk.Button(logs_pane, text="Auto Edit", command=self._start_auto_edit)
        self.run_button.pack(fill="x", pady=(8, 10))

        self.batch_button = ttk.Button(logs_pane, text="Batch Auto Edit", command=self._start_batch_auto_edit)
        self.batch_button.pack(fill="x", pady=(0, 10))

        logs_label = ttk.Label(logs_pane, text="Pipeline Log")
        logs_label.pack(anchor="w")

        log_frame = ttk.Frame(logs_pane)
        log_frame.pack(fill="both", expand=True)

        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical")
        self.log_box = tk.Text(
            log_frame,
            height=14,
            wrap="word",
            state="disabled",
            yscrollcommand=log_scrollbar.set,
        )
        log_scrollbar.configure(command=self.log_box.yview)
        self.log_box.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

    def _bind_mousewheel_scrolling(self, canvas: tk.Canvas, frame: ttk.Frame) -> None:
        def bind_children(widget: tk.Misc) -> None:
            widget.bind("<Enter>", lambda event: self.root.bind_all("<MouseWheel>", on_mousewheel))
            widget.bind("<Leave>", lambda event: self.root.unbind_all("<MouseWheel>"))
            for child in widget.winfo_children():
                bind_children(child)

        def on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-event.delta / 120), "units")

        bind_children(frame)

    def _path_row(
        self,
        parent: ttk.Frame,
        label: str,
        text_var: tk.StringVar,
        browse_command,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)

        ttk.Label(row, text=label, width=20).pack(side="left")
        ttk.Entry(row, textvariable=text_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse", command=browse_command).pack(side="left")

    def _entry_row(
        self,
        parent: ttk.Frame,
        label: str,
        text_var: tk.StringVar,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)

        ttk.Label(row, text=label, width=20).pack(side="left")
        ttk.Entry(row, textvariable=text_var).pack(side="left", fill="x", expand=True)

    def _pick_voiceover(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Voiceover",
            initialdir=str(Path.cwd()),
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg")],
        )
        if selected:
            self.voiceover_var.set(selected)

    def _pick_clips_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select Clips Folder", initialdir=str(Path.cwd()))
        if selected:
            self.clips_var.set(selected)

    def _pick_music_source(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Music File (or Cancel to choose folder)",
            initialdir=str(Path.cwd()),
            filetypes=[("Audio Files", AUDIO_FILE_TYPES)],
        )
        if not selected:
            selected = filedialog.askdirectory(title="Select Music Folder", initialdir=str(Path.cwd()))
        if selected:
            self.music_var.set(selected)

    def _pick_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select Output MP4",
            initialfile=Path(self.output_var.get() or "edited_video.mp4").name,
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
        )
        if selected:
            self.output_var.set(selected)

    def _pick_batch_voiceovers_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select Voiceovers Folder", initialdir=str(Path.cwd()))
        if selected:
            self.batch_voiceovers_var.set(selected)

    def _pick_batch_manifest(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Batch Manifest CSV",
            initialdir=str(Path.cwd()),
            filetypes=[("CSV Files", "*.csv")],
        )
        if selected:
            self.batch_manifest_var.set(selected)

    def _pick_batch_output_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select Batch Output Folder", initialdir=str(Path.cwd()))
        if selected:
            self.batch_output_var.set(selected)

    def _pick_logo(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Logo PNG",
            initialdir=str(Path.cwd()),
            filetypes=[("PNG Image", "*.png")],
        )
        if selected:
            self.logo_path_var.set(selected)

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                message = self._log_queue.get_nowait()
                self._append_log(message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _collect_request(self) -> AutoEditRequest:
        voiceover_raw = self.voiceover_var.get().strip()
        voiceover = Path(voiceover_raw) if voiceover_raw else None
        clips_raw = self.clips_var.get().strip()
        clips_folder = Path(clips_raw) if clips_raw else None
        output = Path(self.output_var.get().strip())
        music_raw = self.music_var.get().strip()
        music = Path(music_raw) if music_raw else None
        logo_raw = self.logo_path_var.get().strip()
        logo = Path(logo_raw) if logo_raw else None
        script_text = self.script_text_widget.get("1.0", "end").strip()
        script_voice = self.script_voice_var.get().strip()

        if not script_text and (not voiceover or not voiceover.exists() or not voiceover.is_file()):
            raise ValueError("Provide a Voiceover file OR paste a Script for TTS.")
        if voiceover and voiceover_raw and (not voiceover.exists() or not voiceover.is_file()):
            raise ValueError("Voiceover file does not exist.")
        if clips_folder and (not clips_folder.exists() or not clips_folder.is_dir()):
            raise ValueError("Clips folder must exist if provided.")
        if music and (not music.exists() or not (music.is_dir() or music.is_file())):
            raise ValueError("Music source must be an existing file or folder.")
        if music and music.is_file() and music.suffix.lower() not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
            raise ValueError("Music file must be a supported audio type.")
        if logo and (not logo.exists() or not logo.is_file()):
            raise ValueError("Logo file does not exist.")
        if logo and logo.suffix.lower() != ".png":
            raise ValueError("Logo must be a .png file")
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output file must end with .mp4")
        if not clips_folder and not self.allow_stock_fetch_var.get():
            raise ValueError("Provide a clips folder or enable stock footage fetching.")

        return AutoEditRequest(
            voiceover_path=voiceover or Path(""),
            clips_folder=clips_folder,
            output_path=output,
            music_folder=music,
            output_width=self.width_var.get(),
            output_height=self.height_var.get(),
            fps=self.fps_var.get(),
            render_preset=RENDER_SPEED_PRESETS.get(self.render_speed_var.get(), "veryfast"),
            allow_stock_fetch=self.allow_stock_fetch_var.get(),
            stock_keywords=self.stock_keywords_var.get().strip(),
            transition_style=TRANSITION_STYLE_MAP.get(self.transition_style_var.get(), "pro_weighted"),
            transition_duration=float(self.transition_duration_var.get()),
            caption_style=CAPTION_STYLE_MAP.get(self.caption_style_var.get(), "bold_stroke"),
            whisper_model=self.whisper_model_var.get() or "base",
            caption_position_ratio=float(self.caption_position_ratio_var.get()),
            caption_max_lines=int(self.caption_max_lines_var.get()),
            caption_font_scale=float(self.caption_font_scale_var.get()),
            caption_pop_scale=float(self.caption_pop_scale_var.get()),
            enable_adaptive_caption_safe_zones=self.adaptive_safe_zones_var.get(),
            enable_karaoke_highlight=self.karaoke_highlight_var.get(),
            enable_motion_overlays=self.enable_motion_overlays_var.get(),
            stat_badge_text=self.stat_badge_text_var.get().strip(),
            cta_text=self.cta_text_var.get().strip(),
            logo_path=logo,
            enable_progress_bar=self.enable_progress_bar_var.get(),
            script_text=script_text,
            script_voice=script_voice,
        )

    def _start_auto_edit(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("In Progress", "Auto edit is already running.")
            return

        try:
            request = self._collect_request()
        except ValueError as exc:
            messagebox.showwarning("Invalid Input", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self._append_log("Starting auto-edit pipeline...")

        def worker() -> None:
            try:
                run_auto_edit(request, log=lambda msg: self._log_queue.put(msg))
                self.root.after(0, self._on_success, str(request.output_path))
            except Exception as exc:
                self.root.after(0, self._on_error, str(exc))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _collect_batch_base_request(self) -> AutoEditRequest:
        clips_raw = self.clips_var.get().strip()
        clips_folder = Path(clips_raw) if clips_raw else None
        music_raw = self.music_var.get().strip()
        music = Path(music_raw) if music_raw else None
        logo_raw = self.logo_path_var.get().strip()
        logo = Path(logo_raw) if logo_raw else None

        if clips_folder and (not clips_folder.exists() or not clips_folder.is_dir()):
            raise ValueError("Clips folder must exist if provided.")
        if music and (not music.exists() or not (music.is_dir() or music.is_file())):
            raise ValueError("Music source must be an existing file or folder.")
        if music and music.is_file() and music.suffix.lower() not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
            raise ValueError("Music file must be a supported audio type.")
        if logo and (not logo.exists() or not logo.is_file()):
            raise ValueError("Logo file does not exist.")
        if logo and logo.suffix.lower() != ".png":
            raise ValueError("Logo must be a .png file")
        if not clips_folder and not self.allow_stock_fetch_var.get():
            raise ValueError("Provide a clips folder or enable stock footage fetching.")

        return AutoEditRequest(
            voiceover_path=Path(""),
            clips_folder=clips_folder,
            output_path=Path("output/batch_placeholder.mp4"),
            music_folder=music,
            output_width=self.width_var.get(),
            output_height=self.height_var.get(),
            fps=self.fps_var.get(),
            render_preset=RENDER_SPEED_PRESETS.get(self.render_speed_var.get(), "veryfast"),
            allow_stock_fetch=self.allow_stock_fetch_var.get(),
            stock_keywords=self.stock_keywords_var.get().strip(),
            transition_style=TRANSITION_STYLE_MAP.get(self.transition_style_var.get(), "pro_weighted"),
            transition_duration=float(self.transition_duration_var.get()),
            caption_style=CAPTION_STYLE_MAP.get(self.caption_style_var.get(), "bold_stroke"),
            whisper_model=self.whisper_model_var.get() or "base",
            caption_position_ratio=float(self.caption_position_ratio_var.get()),
            caption_max_lines=int(self.caption_max_lines_var.get()),
            caption_font_scale=float(self.caption_font_scale_var.get()),
            caption_pop_scale=float(self.caption_pop_scale_var.get()),
            enable_adaptive_caption_safe_zones=self.adaptive_safe_zones_var.get(),
            enable_karaoke_highlight=self.karaoke_highlight_var.get(),
            enable_motion_overlays=self.enable_motion_overlays_var.get(),
            stat_badge_text=self.stat_badge_text_var.get().strip(),
            cta_text=self.cta_text_var.get().strip(),
            logo_path=logo,
            enable_progress_bar=self.enable_progress_bar_var.get(),
            script_text="",
            script_voice="",
        )

    def _start_batch_auto_edit(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("In Progress", "A render job is already running.")
            return

        voiceovers_raw = self.batch_voiceovers_var.get().strip()
        output_raw = self.batch_output_var.get().strip()
        manifest_raw = self.batch_manifest_var.get().strip()
        if not voiceovers_raw:
            messagebox.showwarning("Invalid Input", "Select a Voiceovers Folder for batch mode.")
            return
        if not output_raw:
            messagebox.showwarning("Invalid Input", "Select a Batch Output folder.")
            return

        voiceovers_folder = Path(voiceovers_raw)
        output_folder = Path(output_raw)
        manifest_path = Path(manifest_raw) if manifest_raw else None

        if not voiceovers_folder.exists() or not voiceovers_folder.is_dir():
            messagebox.showwarning("Invalid Input", "Voiceovers Folder does not exist.")
            return
        if manifest_path and (not manifest_path.exists() or not manifest_path.is_file()):
            messagebox.showwarning("Invalid Input", "Manifest CSV file does not exist.")
            return

        try:
            base_request = self._collect_batch_base_request()
        except ValueError as exc:
            messagebox.showwarning("Invalid Input", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.batch_button.configure(state="disabled")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self._append_log("Starting batch pipeline...")

        def worker() -> None:
            try:
                summary = run_batch_auto_edit(
                    base_request=base_request,
                    voiceovers_folder=voiceovers_folder,
                    output_folder=output_folder,
                    manifest_path=manifest_path,
                    log=lambda msg: self._log_queue.put(msg),
                )
                self.root.after(0, self._on_batch_success, summary, str(output_folder))
            except Exception as exc:
                self.root.after(0, self._on_error, str(exc))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _on_success(self, output_path: str) -> None:
        self.run_button.configure(state="normal")
        self.batch_button.configure(state="normal")
        self._append_log("Done.")
        messagebox.showinfo("Export Complete", f"Output saved to:\n{output_path}")

    def _on_batch_success(self, summary: dict[str, int], output_folder: str) -> None:
        self.run_button.configure(state="normal")
        self.batch_button.configure(state="normal")
        self._append_log("Batch done.")
        messagebox.showinfo(
            "Batch Complete",
            f"Output folder: {output_folder}\n"
            f"Total: {summary.get('total', 0)}\n"
            f"Success: {summary.get('success', 0)}\n"
            f"Failed: {summary.get('failed', 0)}",
        )

    def _on_error(self, error: str) -> None:
        self.run_button.configure(state="normal")
        self.batch_button.configure(state="normal")
        self._append_log(f"Error: {error}")
        messagebox.showerror("Auto Edit Failed", error)


def run_app() -> None:
    root = tk.Tk()
    AutoEditorApp(root)
    root.mainloop()
