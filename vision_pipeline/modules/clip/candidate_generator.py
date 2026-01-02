from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPCandidateGenerator:
    """Generate top-K label candidates using CLIP cosine similarity."""

    def __init__(self, config: dict):
        if config is None:
            config = {}
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.model_name = config.get("model_name", "openai/clip-vit-base-patch32")
        self.device = config.get("device", "auto")
        self.top_k = int(config.get("top_k", 5))
        self.top1_threshold = float(config.get("top1_threshold", 0.55))
        self.prompt_templates = config.get(
            "prompt_templates",
            [
                "a plush toy of {label}{description_clause}",
                "a stuffed doll representing {label}{description_clause}",
            ],
        )
        self.text_batch_size = int(config.get("text_batch_size", 64))
        self.cache_path = config.get("cache_path", "data/artifacts/clip_text_embeddings.pt")
        self.label_entries = self._load_label_entries(config)
        self.labels = [entry["label"] for entry in self.label_entries]
        self._lock = Lock()
        self._available = False

        if not self.enabled:
            return
        if not self.labels:
            print("[CLIPCandidateGenerator] No labels configured. Disabling CLIP candidates.")
            return

        self._resolve_device()
        print(f"[CLIPCandidateGenerator] Loading CLIP model: {self.model_name} on {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            print(f"[CLIPCandidateGenerator] Failed to load CLIP model: {e}")
            return

        try:
            self.text_embeddings = self._load_or_build_text_embeddings()
        except Exception as e:
            print(f"[CLIPCandidateGenerator] Failed to build text embeddings: {e}")
            return

        self._available = True
        print(f"[CLIPCandidateGenerator] Ready with {len(self.labels)} labels.")

    def is_available(self) -> bool:
        return self._available

    def _resolve_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("[CLIPCandidateGenerator] CUDA not available, falling back to MPS")
                self.device = "mps"
            else:
                print("[CLIPCandidateGenerator] CUDA not available, falling back to CPU")
                self.device = "cpu"

        if self.device == "mps" and not torch.backends.mps.is_available():
            if torch.cuda.is_available():
                print("[CLIPCandidateGenerator] MPS not available, falling back to CUDA")
                self.device = "cuda"
            else:
                print("[CLIPCandidateGenerator] MPS not available, falling back to CPU")
                self.device = "cpu"

    def _load_label_entries(self, config: dict) -> list[dict]:
        labels = config.get("labels")
        labels_path = config.get("labels_path")
        if labels_path:
            project_root = Path(__file__).resolve().parents[2]
            path = Path(labels_path)
            if not path.is_absolute():
                path = project_root / path
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "labels" in data:
                    labels = data.get("labels")
                else:
                    labels = data
            except Exception as e:
                print(f"[CLIPCandidateGenerator] Failed to load labels from {path}: {e}")
                labels = None
        return self._normalize_label_entries(labels)

    def _normalize_label_entries(self, labels) -> list[dict]:
        entries = []
        if not labels:
            return entries

        if isinstance(labels, dict):
            for label, description in labels.items():
                entry = self._build_label_entry(label, description)
                if entry:
                    entries.append(entry)
            return entries

        if isinstance(labels, list):
            for item in labels:
                if isinstance(item, str):
                    entry = self._build_label_entry(item, None)
                elif isinstance(item, dict):
                    if "label" in item:
                        entry = self._build_label_entry(item.get("label"), item.get("description"))
                    elif len(item) == 1:
                        label, description = next(iter(item.items()))
                        entry = self._build_label_entry(label, description)
                    else:
                        entry = None
                else:
                    entry = None
                if entry:
                    entries.append(entry)
        return entries

    def _build_label_entry(self, label, description) -> Optional[dict]:
        if label is None:
            return None
        label_str = str(label).strip()
        if not label_str:
            return None
        desc_str = ""
        if description is not None:
            desc_str = str(description).strip()
        return {"label": label_str, "description": desc_str}

    def _format_prompt(self, template: str, label: str, description: Optional[str] = None) -> str:
        description = (description or "").strip()
        description_clause = f", {description}" if description else ""
        try:
            rendered = template.format(
                character_name=label,
                label=label,
                description=description,
                description_clause=description_clause,
            )
        except KeyError:
            rendered = template
            rendered = rendered.replace("{character_name}", label)
            rendered = rendered.replace("{label}", label)
            rendered = rendered.replace("{description_clause}", description_clause)
            rendered = rendered.replace("{description}", description)
        if not description:
            rendered = rendered.rstrip(" ,")
        return rendered

    def _load_or_build_text_embeddings(self) -> torch.Tensor:
        cache_path = self._resolve_cache_path()
        if cache_path and cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu")
                cached_entries = cached.get("label_entries")
                cached_labels = cached.get("labels")
                entries_match = False
                if cached_entries is not None:
                    entries_match = cached_entries == self.label_entries
                elif cached_labels is not None:
                    entries_match = cached_labels == self.labels and all(
                        not entry.get("description") for entry in self.label_entries
                    )

                if (
                    cached.get("model_name") == self.model_name
                    and cached.get("prompt_templates") == self.prompt_templates
                    and entries_match
                ):
                    embeddings = cached.get("embeddings")
                    if isinstance(embeddings, torch.Tensor):
                        return embeddings.to(self.device)
            except Exception as e:
                print(f"[CLIPCandidateGenerator] Cache load failed, rebuilding: {e}")

        embeddings = self._build_text_embeddings()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": self.model_name,
                    "prompt_templates": self.prompt_templates,
                    "labels": self.labels,
                    "label_entries": self.label_entries,
                    "embeddings": embeddings.detach().cpu(),
                },
                cache_path,
            )
        return embeddings

    def _resolve_cache_path(self) -> Optional[Path]:
        if not self.cache_path:
            return None
        project_root = Path(__file__).resolve().parents[2]
        path = Path(self.cache_path)
        if not path.is_absolute():
            path = project_root / path
        return path

    def _build_text_embeddings(self) -> torch.Tensor:
        prompts = []
        prompt_counts = []
        for entry in self.label_entries:
            label = entry["label"]
            description = entry.get("description")
            label_prompts = [self._format_prompt(t, label, description) for t in self.prompt_templates]
            prompts.extend(label_prompts)
            prompt_counts.append(len(label_prompts))

        text_embeddings = []
        for start in range(0, len(prompts), self.text_batch_size):
            batch_prompts = prompts[start:start + self.text_batch_size]
            inputs = self.processor(text=batch_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            features = F.normalize(features, p=2, dim=-1)
            text_embeddings.append(features.detach().cpu())

        all_embeddings = torch.cat(text_embeddings, dim=0)
        label_embeddings = []
        offset = 0
        for count in prompt_counts:
            group = all_embeddings[offset:offset + count]
            label_emb = group.mean(dim=0)
            label_embeddings.append(label_emb)
            offset += count

        stacked = torch.stack(label_embeddings, dim=0)
        stacked = F.normalize(stacked, p=2, dim=-1)
        return stacked.to(self.device)

    def get_candidates(self, image: Image.Image, top_k: Optional[int] = None) -> tuple[list[dict], Optional[float]]:
        if not self._available:
            return [], None
        if top_k is None:
            top_k = self.top_k
        if top_k <= 0:
            return [], None

        with self._lock:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            similarities = image_features @ self.text_embeddings.T

        similarities = similarities.squeeze(0)
        k = min(top_k, similarities.shape[0])
        values, indices = torch.topk(similarities, k=k)

        candidates = []
        for score, index in zip(values.tolist(), indices.tolist()):
            candidates.append({"label": self.labels[index], "score": float(score)})

        top1_score = candidates[0]["score"] if candidates else None
        return candidates, top1_score
