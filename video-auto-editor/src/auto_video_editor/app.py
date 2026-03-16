from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .models import AutoEditRequest
from .orchestrator import run_auto_edit

RENDER_SPEED_PRESETS = {
    "Fast": "veryfast",
    "Balanced": "medium",
    "Quality": "slow",
}

TRANSITION_STYLE_MAP = {
    "None": "none",
    "Crossfade": "crossfade",
    "Zoom": "zoom",
    "Fade Black": "fade_black",
}

CAPTION_STYLE_MAP = {
    "Bold Stroke": "bold_stroke",
    "Yellow Active": "yellow_active",
    "Gradient Fill": "gradient_fill",
}


class AutoEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Local Auto Video Editor")
        self.root.geometry("980x700")

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
        self.transition_style_var = tk.StringVar(value="Crossfade")
        self.transition_duration_var = tk.DoubleVar(value=0.22)
        self.caption_style_var = tk.StringVar(value="Bold Stroke")
        self.whisper_model_var = tk.StringVar(value="base")
        self.caption_position_ratio_var = tk.DoubleVar(value=0.54)
        self.caption_max_lines_var = tk.IntVar(value=3)
        self.caption_font_scale_var = tk.DoubleVar(value=1.00)
        self.caption_pop_scale_var = tk.DoubleVar(value=1.00)
        self.adaptive_safe_zones_var = tk.BooleanVar(value=True)
        self.karaoke_highlight_var = tk.BooleanVar(value=True)

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        self._path_row(
            frame,
            label="Voiceover",
            text_var=self.voiceover_var,
            browse_command=self._pick_voiceover,
        )
        self._path_row(
            frame,
            label="Clips Folder (optional)",
            text_var=self.clips_var,
            browse_command=self._pick_clips_folder,
        )
        self._entry_row(
            frame,
            label="Stock Search",
            text_var=self.stock_keywords_var,
        )
        self._path_row(
            frame,
            label="Music Folder",
            text_var=self.music_var,
            browse_command=self._pick_music_folder,
        )
        self._path_row(
            frame,
            label="Output File",
            text_var=self.output_var,
            browse_command=self._pick_output,
        )

        settings = ttk.LabelFrame(frame, text="Render Settings", padding=8)
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
            text="Fetch stock clips from Pexels/Pixabay if local clips are missing",
            variable=self.allow_stock_fetch_var,
        ).grid(row=2, column=0, columnspan=6, padx=8, pady=(2, 6), sticky="w")

        ttk.Label(settings, text="Transition").grid(row=3, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.transition_style_var,
            values=["None", "Crossfade", "Zoom", "Fade Black"],
            state="readonly",
            width=12,
        ).grid(row=3, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(settings, text="Trans. Dur (s)").grid(row=3, column=2, padx=8, pady=6, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.1,
            to=2.0,
            increment=0.1,
            textvariable=self.transition_duration_var,
            width=8,
            format="%.1f",
        ).grid(row=3, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(settings, text="Captions").grid(row=4, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.caption_style_var,
            values=list(CAPTION_STYLE_MAP.keys()),
            state="readonly",
            width=12,
        ).grid(row=4, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(settings, text="Whisper").grid(row=4, column=2, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.whisper_model_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
            width=10,
        ).grid(row=4, column=3, padx=8, pady=6, sticky="w")

        subtitle_style = ttk.LabelFrame(frame, text="Subtitle Styling Editor", padding=8)
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

        self.run_button = ttk.Button(frame, text="Auto Edit", command=self._start_auto_edit)
        self.run_button.pack(fill="x", pady=(8, 10))

        logs_label = ttk.Label(frame, text="Pipeline Log")
        logs_label.pack(anchor="w")

        self.log_box = tk.Text(frame, height=20, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)

    def _path_row(
        self,
        parent: ttk.Frame,
        label: str,
        text_var: tk.StringVar,
        browse_command,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)

        ttk.Label(row, text=label, width=12).pack(side="left")
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

        ttk.Label(row, text=label, width=12).pack(side="left")
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

    def _pick_music_folder(self) -> None:
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
        voiceover = Path(self.voiceover_var.get().strip())
        clips_raw = self.clips_var.get().strip()
        clips_folder = Path(clips_raw) if clips_raw else None
        output = Path(self.output_var.get().strip())
        music_raw = self.music_var.get().strip()
        music = Path(music_raw) if music_raw else None

        if not voiceover.exists() or not voiceover.is_file():
            raise ValueError("Voiceover file is required and must exist.")
        if clips_folder and (not clips_folder.exists() or not clips_folder.is_dir()):
            raise ValueError("Clips folder must exist if provided.")
        if music and (not music.exists() or not music.is_dir()):
            raise ValueError("Music folder does not exist.")
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output file must end with .mp4")
        if not clips_folder and not self.allow_stock_fetch_var.get():
            raise ValueError("Provide a clips folder or enable stock footage fetching.")

        return AutoEditRequest(
            voiceover_path=voiceover,
            clips_folder=clips_folder,
            output_path=output,
            music_folder=music,
            output_width=self.width_var.get(),
            output_height=self.height_var.get(),
            fps=self.fps_var.get(),
            render_preset=RENDER_SPEED_PRESETS.get(self.render_speed_var.get(), "veryfast"),
            allow_stock_fetch=self.allow_stock_fetch_var.get(),
            stock_keywords=self.stock_keywords_var.get().strip(),
            transition_style=TRANSITION_STYLE_MAP.get(self.transition_style_var.get(), "crossfade"),
            transition_duration=float(self.transition_duration_var.get()),
            caption_style=CAPTION_STYLE_MAP.get(self.caption_style_var.get(), "beast"),
            whisper_model=self.whisper_model_var.get() or "base",
            caption_position_ratio=float(self.caption_position_ratio_var.get()),
            caption_max_lines=int(self.caption_max_lines_var.get()),
            caption_font_scale=float(self.caption_font_scale_var.get()),
            caption_pop_scale=float(self.caption_pop_scale_var.get()),
            enable_adaptive_caption_safe_zones=self.adaptive_safe_zones_var.get(),
            enable_karaoke_highlight=self.karaoke_highlight_var.get(),
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

    def _on_success(self, output_path: str) -> None:
        self.run_button.configure(state="normal")
        self._append_log("Done.")
        messagebox.showinfo("Export Complete", f"Output saved to:\n{output_path}")

    def _on_error(self, error: str) -> None:
        self.run_button.configure(state="normal")
        self._append_log(f"Error: {error}")
        messagebox.showerror("Auto Edit Failed", error)


def run_app() -> None:
    root = tk.Tk()
    AutoEditorApp(root)
    root.mainloop()
