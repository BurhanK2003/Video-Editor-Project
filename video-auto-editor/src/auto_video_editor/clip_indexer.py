from __future__ import annotations

import pickle
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IGNORED_PATH_PARTS = {"_stock_cache", "_clip_cache", "__pycache__", "batch_logs", "async_online"}


@dataclass
class SceneEmbedding:
    clip_path: Path
    scene_start: float
    scene_end: float
    embedding: np.ndarray
    best_window_start: float
    best_window_end: float


class ClipIndexer:
    def __init__(
        self,
        clips_folder: Path,
        model_name: str = "ViT-B/32",
        cache_file: Path | None = None,
        log: callable | None = None,
    ) -> None:
        self.clips_folder = clips_folder
        self.model_name = model_name
        self.model_id = self._resolve_model_id(model_name)
        self.cache_file = cache_file or self._cache_file_for_model(self.model_id)
        self.log = log or (lambda _msg: None)

        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._dtype = None

        self._cache = self._load_cache()

    @staticmethod
    def _resolve_model_id(model_name: str) -> str:
        normalized = (model_name or "").strip().lower()
        if normalized in {"vit-l/14", "vit_l_14", "vit-l14"}:
            return "openai/clip-vit-large-patch14"
        return "openai/clip-vit-base-patch32"

    def _cache_file_for_model(self, model_id: str) -> Path:
        suffix = "vitl14" if "large" in model_id else "vitb32"
        return self.clips_folder / f".clip_scene_cache_{suffix}.pkl"

    def _load_cache(self) -> dict:
        if not self.cache_file.exists():
            return {"version": 1, "model_id": self.model_id, "clips": {}}
        try:
            with self.cache_file.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                return {"version": 1, "model_id": self.model_id, "clips": {}}
            if payload.get("model_id") != self.model_id:
                return {"version": 1, "model_id": self.model_id, "clips": {}}
            if "clips" not in payload or not isinstance(payload["clips"], dict):
                payload["clips"] = {}
            return payload
        except Exception:
            return {"version": 1, "model_id": self.model_id, "clips": {}}

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("wb") as handle:
            pickle.dump(self._cache, handle)

    def _load_clip_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        from transformers import CLIPModel, CLIPProcessor
        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        # ViT-L/14 is significantly slower on CPU. Prefer ViT-B/32 unless explicitly forced.
        allow_large_on_cpu = os.getenv("CLIP_ALLOW_LARGE_ON_CPU", "0").strip().lower() in {"1", "true", "yes"}
        if self._device == "cpu" and self.model_id == "openai/clip-vit-large-patch14" and not allow_large_on_cpu:
            prev_model = self.model_id
            self.model_id = "openai/clip-vit-base-patch32"
            self.log(
                "CLIP runtime optimization: switching model "
                f"{prev_model} -> {self.model_id} on CPU "
                "(set CLIP_ALLOW_LARGE_ON_CPU=1 to force large model)."
            )
            self.cache_file = self._cache_file_for_model(self.model_id)
            self._cache = self._load_cache()

        self.log(f"CLIP indexer loading model: {self.model_id} on {self._device}")
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._model.to(self._device)
        self._model.eval()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def _coerce_feature_tensor(self, raw_output, *, projection_name: str | None = None):
        """Normalize model outputs across transformers versions to a torch tensor."""
        assert self._torch is not None

        tensor = raw_output
        if not self._torch.is_tensor(tensor):
            if hasattr(tensor, "text_embeds") and tensor.text_embeds is not None:
                tensor = tensor.text_embeds
            elif hasattr(tensor, "image_embeds") and tensor.image_embeds is not None:
                tensor = tensor.image_embeds
            elif hasattr(tensor, "pooler_output") and tensor.pooler_output is not None:
                tensor = tensor.pooler_output
            elif hasattr(tensor, "last_hidden_state") and tensor.last_hidden_state is not None:
                tensor = tensor.last_hidden_state.mean(dim=1)
            elif isinstance(tensor, (list, tuple)) and tensor:
                tensor = tensor[0]

        if not self._torch.is_tensor(tensor):
            raise TypeError("Unsupported CLIP feature output type")

        if projection_name and self._model is not None and hasattr(self._model, projection_name):
            projection = getattr(self._model, projection_name)
            try:
                tensor = projection(tensor)
            except Exception:
                # Projection may already be applied for some model paths.
                pass

        tensor = tensor.to(dtype=self._dtype)
        tensor = tensor / tensor.norm(dim=-1, keepdim=True)
        return tensor

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self._load_clip_model()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        cleaned = [t.strip() or "nature scene" for t in texts]
        with self._torch.no_grad():
            tokens = self._processor(
                text=cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            try:
                feat_raw = self._model.get_text_features(**tokens)
                feat = self._coerce_feature_tensor(feat_raw)
            except Exception:
                text_tokens = {k: v for k, v in tokens.items() if k in {"input_ids", "attention_mask", "position_ids"}}
                if hasattr(self._model, "text_model"):
                    feat_raw = self._model.text_model(**text_tokens)
                    feat = self._coerce_feature_tensor(feat_raw, projection_name="text_projection")
                else:
                    raise
        out = feat.detach().cpu().numpy().astype(np.float32)
        return self._normalize(out)

    def encode_images(self, rgb_images: list[np.ndarray]) -> np.ndarray:
        self._load_clip_model()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        if not rgb_images:
            return np.zeros((0, 512), dtype=np.float32)

        with self._torch.no_grad():
            tensors = self._processor(images=rgb_images, return_tensors="pt")
            pixel_values = tensors["pixel_values"].to(self._device)
            try:
                feat_raw = self._model.get_image_features(pixel_values=pixel_values)
                feat = self._coerce_feature_tensor(feat_raw)
            except Exception:
                if hasattr(self._model, "vision_model"):
                    feat_raw = self._model.vision_model(pixel_values=pixel_values)
                    feat = self._coerce_feature_tensor(feat_raw, projection_name="visual_projection")
                else:
                    raise
        out = feat.detach().cpu().numpy().astype(np.float32)
        return self._normalize(out)

    def list_local_clips(self) -> list[Path]:
        if not self.clips_folder.exists():
            return []
        return [
            p
            for p in sorted(self.clips_folder.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
            and not any(part in IGNORED_PATH_PARTS for part in p.parts)
        ]

    def _video_duration(self, clip_path: Path) -> float:
        import cv2

        cap = cv2.VideoCapture(str(clip_path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps <= 0.0:
                return 0.0
            return max(0.0, frame_count / fps)
        finally:
            cap.release()

    def _detect_scenes(self, clip_path: Path) -> list[tuple[float, float]]:
        try:
            from scenedetect import ContentDetector, SceneManager, open_video
        except Exception:
            duration = self._video_duration(clip_path)
            return [(0.0, duration)] if duration > 0 else []

        try:
            video = open_video(str(clip_path))
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=27.0))
            manager.detect_scenes(video)
            scene_list = manager.get_scene_list()
            scenes: list[tuple[float, float]] = []
            for start_tc, end_tc in scene_list:
                start = float(start_tc.get_seconds())
                end = float(end_tc.get_seconds())
                if end - start >= 0.35:
                    scenes.append((start, end))
            if scenes:
                return scenes
        except Exception:
            pass

        duration = self._video_duration(clip_path)
        return [(0.0, duration)] if duration > 0 else []

    def _frame_at(self, clip_path: Path, timestamp: float) -> np.ndarray | None:
        import cv2

        cap = cv2.VideoCapture(str(clip_path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb
        finally:
            cap.release()

    def _sample_scene_frames(self, clip_path: Path, scene_start: float, scene_end: float) -> list[np.ndarray]:
        dur = max(0.01, scene_end - scene_start)
        points = [scene_start + 0.02 * dur, scene_start + 0.50 * dur, scene_start + 0.98 * dur]
        images: list[np.ndarray] = []
        for ts in points:
            frame = self._frame_at(clip_path, ts)
            if frame is not None:
                images.append(frame)
        return images

    def _sharpness_score(self, gray_frame: np.ndarray) -> float:
        import cv2

        return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

    def _motion_score(self, prev_gray: np.ndarray | None, gray_frame: np.ndarray) -> float:
        if prev_gray is None:
            return 0.5
        import cv2

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(np.mean(mag))

        # Prefer moderate motion around 8 px/frame, penalize both still and shaky footage.
        center = 8.0
        spread = 6.0
        score = math.exp(-((mean_mag - center) ** 2) / max(1e-6, 2 * spread * spread))
        return float(max(0.0, min(1.0, score)))

    def _exposure_score(self, gray_frame: np.ndarray) -> float:
        hist = np.histogram(gray_frame, bins=64, range=(0, 255))[0].astype(np.float32)
        hist = hist / max(1e-6, float(hist.sum()))
        cdf = np.cumsum(hist)
        low_idx = int(np.searchsorted(cdf, 0.05))
        high_idx = int(np.searchsorted(cdf, 0.95))
        spread = max(0, high_idx - low_idx)
        return float(max(0.0, min(1.0, spread / 42.0)))

    def _best_quality_window(self, clip_path: Path, scene_start: float, scene_end: float) -> tuple[float, float]:
        fast_quality = os.getenv("CLIP_INDEXER_FAST", "1").strip().lower() in {"1", "true", "yes"}
        if fast_quality:
            # Fast mode: center window avoids costly frame-by-frame quality analysis.
            window_seconds = 3.0
            scene_duration = max(0.0, scene_end - scene_start)
            if scene_duration <= 0.5:
                return scene_start, min(scene_end, scene_start + max(0.5, scene_duration))
            start = scene_start + max(0.0, (scene_duration - window_seconds) * 0.5)
            end = min(scene_end, start + min(window_seconds, scene_duration))
            if end <= start:
                end = min(scene_end, start + 1.0)
            return float(start), float(end)

        import cv2

        fps_sample = 2.0
        window_seconds = 3.0
        scene_duration = max(0.0, scene_end - scene_start)
        if scene_duration <= 0.5:
            return scene_start, min(scene_end, scene_start + max(0.5, scene_duration))

        timestamps = np.arange(scene_start, scene_end, 1.0 / fps_sample).tolist()
        if not timestamps:
            return scene_start, min(scene_end, scene_start + min(window_seconds, scene_duration))

        scores: list[float] = []
        prev_gray = None
        sharp_values: list[float] = []

        # First pass for robust sharpness scaling.
        grays: list[np.ndarray] = []
        for ts in timestamps:
            frame = self._frame_at(clip_path, ts)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            grays.append(gray)
            sharp_values.append(self._sharpness_score(gray))

        if not grays:
            return scene_start, min(scene_end, scene_start + min(window_seconds, scene_duration))

        s_min = float(min(sharp_values))
        s_max = float(max(sharp_values))
        s_spread = max(1e-6, s_max - s_min)

        for gray in grays:
            sharp = (self._sharpness_score(gray) - s_min) / s_spread
            motion = self._motion_score(prev_gray, gray)
            exposure = self._exposure_score(gray)
            combined = 0.45 * sharp + 0.35 * motion + 0.20 * exposure
            scores.append(float(combined))
            prev_gray = gray

        window_len = max(2, int(round(window_seconds * fps_sample)))
        if len(scores) <= window_len:
            return scene_start, min(scene_end, scene_start + min(window_seconds, scene_duration))

        rolling_sum = sum(scores[:window_len])
        best_avg = rolling_sum / window_len
        best_idx = 0
        for idx in range(window_len, len(scores)):
            rolling_sum += scores[idx] - scores[idx - window_len]
            avg = rolling_sum / window_len
            if avg > best_avg:
                best_avg = avg
                best_idx = idx - window_len + 1

        best_start = min(scene_end - 0.2, scene_start + best_idx * (1.0 / fps_sample))
        best_end = min(scene_end, best_start + window_seconds)
        if best_end <= best_start:
            best_end = min(scene_end, best_start + 1.0)
        return float(best_start), float(best_end)

    def _clip_cache_key(self, clip_path: Path) -> str:
        return str(clip_path.resolve())

    def index_clip(self, clip_path: Path) -> list[SceneEmbedding]:
        key = self._clip_cache_key(clip_path)
        stat = clip_path.stat()
        mtime = float(stat.st_mtime)

        cache_item = self._cache["clips"].get(key)
        if isinstance(cache_item, dict) and float(cache_item.get("mtime", -1.0)) == mtime:
            scenes_payload = cache_item.get("scenes") or []
            restored: list[SceneEmbedding] = []
            for scene in scenes_payload:
                emb = np.asarray(scene.get("embedding") or [], dtype=np.float32)
                if emb.size == 0:
                    continue
                restored.append(
                    SceneEmbedding(
                        clip_path=clip_path,
                        scene_start=float(scene.get("scene_start", 0.0)),
                        scene_end=float(scene.get("scene_end", 0.0)),
                        embedding=emb,
                        best_window_start=float(scene.get("best_window_start", scene.get("scene_start", 0.0))),
                        best_window_end=float(scene.get("best_window_end", scene.get("scene_end", 0.0))),
                    )
                )
            if restored:
                return restored

        scenes = self._detect_scenes(clip_path)
        embedded: list[SceneEmbedding] = []
        for scene_start, scene_end in scenes:
            if scene_end - scene_start < 0.4:
                continue
            frames = self._sample_scene_frames(clip_path, scene_start, scene_end)
            if not frames:
                continue
            frame_embeddings = self.encode_images(frames)
            if frame_embeddings.shape[0] == 0:
                continue
            scene_embedding = self._normalize(np.mean(frame_embeddings, axis=0, keepdims=True))[0]
            best_start, best_end = self._best_quality_window(clip_path, scene_start, scene_end)
            embedded.append(
                SceneEmbedding(
                    clip_path=clip_path,
                    scene_start=float(scene_start),
                    scene_end=float(scene_end),
                    embedding=scene_embedding.astype(np.float32),
                    best_window_start=best_start,
                    best_window_end=best_end,
                )
            )

        self._cache["clips"][key] = {
            "mtime": mtime,
            "scenes": [
                {
                    "scene_start": s.scene_start,
                    "scene_end": s.scene_end,
                    "embedding": s.embedding.tolist(),
                    "best_window_start": s.best_window_start,
                    "best_window_end": s.best_window_end,
                }
                for s in embedded
            ],
        }
        self._save_cache()
        return embedded

    def build_local_scene_index(self, clip_paths: list[Path] | None = None) -> list[SceneEmbedding]:
        clips = clip_paths if clip_paths is not None else self.list_local_clips()
        all_scenes: list[SceneEmbedding] = []
        for path in clips:
            try:
                scenes = self.index_clip(path)
                all_scenes.extend(scenes)
            except Exception as exc:
                self.log(f"Clip indexing failed for {path.name}: {exc}")
        self.log(f"CLIP scene index ready: clips={len(clips)}, scenes={len(all_scenes)}")
        return all_scenes
