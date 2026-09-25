"""Local visual and text encoders; downloads are handled by setup, never inference.

The vision artifact is the author's AffectNet-trained HSEmotion/EmotiEffLib
EfficientNet-B2. Its max pooling and RGB preprocessing are deliberate. The
text artifact is clean DeBERTa-v3-large, without a third-party MELD fine-tune.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


VISION_FEATURE_DIM = 1408
TEXT_FEATURE_DIM = 1024
VISION_PARAMETER_COUNT = 7_710_857
TEXT_PARAMETER_COUNT = 434_012_160
VISION_SHA256 = "e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f"
VISION_REVISION = "520a051c64cd191521e5934655314e769a319684"
TEXT_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
TEXT_SHA256 = "dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5"
FER_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")


def count_parameters(model: nn.Module) -> int:
    """Count unique learned values, including frozen weights; exclude buffers."""
    return sum(parameter.numel() for parameter in model.parameters())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(device: str | torch.device) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the environment check or select CPU explicitly.")
    return result


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _check_features(features: np.ndarray, dimension: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != dimension:
        raise RuntimeError(f"Encoder returned invalid feature shape {features.shape}.")
    if not np.isfinite(features).all():
        raise RuntimeError("Encoder returned nonfinite features; retry without mixed precision.")
    return features


class VisionEncoder(nn.Module):
    """AffectNet-pretrained face features and retained seven-class diagnostics.

    Face detection, box expansion and utterance pooling belong to the pipeline.
    ``encode_faces`` accepts RGB images only, never OpenCV BGR arrays. The
    original classifier remains attached and is included in parameter auditing.
    """

    feature_dim = VISION_FEATURE_DIM
    labels = FER_LABELS

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        batch_size: int = 16,
        mixed_precision: bool = True,
        adapted_state_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.device = _device(device)
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Visual checkpoint not found: {checkpoint_path}")
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != VISION_SHA256:
            raise ValueError("The visual checkpoint does not match the pinned author's artifact.")

        # The author distributes an entire module as a pickle. Only this exact
        # hash-verified artifact can enter the legacy loader. timm==0.9.16
        # supplies the classes referenced in its pickle; arbitrary user-uploaded
        # checkpoint files must never be passed to this path.
        import timm  # noqa: F401

        self.model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(self.model, nn.Module):
            raise TypeError("Expected the pinned EfficientNet module.")
        if getattr(self.model.global_pool, "pool_type", None) != "max":
            raise RuntimeError("The pretrained visual model requires max pooling.")
        if self.model.classifier.in_features != self.feature_dim or self.model.classifier.out_features != 7:
            raise RuntimeError("Unexpected visual classifier dimensions.")
        if count_parameters(self.model) != VISION_PARAMETER_COUNT:
            raise RuntimeError("The visual parameter inventory does not match the pinned architecture.")
        if adapted_state_path is not None:
            self.model.load_state_dict(torch.load(adapted_state_path, map_location="cpu", weights_only=True))
        self.artifact_sha256 = actual_hash
        self.adapted_sha256 = sha256_file(adapted_state_path) if adapted_state_path else None
        self.model.requires_grad_(False)
        self.model.to(self.device).eval()
        # Matches the author's Torch preprocessing (RGB, direct 260x260 resize).
        self.preprocess = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Differentiable feature path for an optional limited MELD fine-tune."""
        return self.model.forward_head(self.model.forward_features(images), pre_logits=True)

    def forward_tensors(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)

    def preprocess_faces(self, faces: Sequence[Image.Image | np.ndarray]) -> torch.Tensor:
        images = []
        for face in faces:
            if isinstance(face, np.ndarray):
                if face.ndim != 3 or face.shape[2] != 3 or face.dtype != np.uint8:
                    raise ValueError("Faces must be uint8 RGB arrays with shape HxWx3.")
                face = Image.fromarray(face, mode="RGB")
            if not isinstance(face, Image.Image):
                raise TypeError("Faces must be RGB NumPy arrays or PIL images.")
            images.append(self.preprocess(face.convert("RGB")))
        if not images:
            return torch.empty((0, 3, 260, 260))
        return torch.stack(images)

    def encode_faces(self, faces: Sequence[Image.Image | np.ndarray]) -> np.ndarray:
        if len(faces) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(faces), self.batch_size):
                pixels = self.preprocess_faces(faces[start:start + self.batch_size]).to(self.device)
                with _autocast(self.device, self.mixed_precision):
                    features = self.forward(pixels)
                outputs.append(features.float().cpu().numpy())
        return _check_features(np.concatenate(outputs), self.feature_dim)

    def fer_logits(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        """AffectNet diagnostics in FER_LABELS order; these are not MELD tags."""
        values = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of shape Nx{self.feature_dim}.")
        with torch.inference_mode():
            return self.model.classifier(values).float().cpu().numpy()

    def parameter_count(self) -> int:
        return count_parameters(self.model)

    def unfreeze_last_layers(self, stages: int = 1) -> int:
        """Unfreeze final feature blocks; callers keep batchnorm statistics fixed."""
        if not 1 <= stages <= len(self.model.blocks):
            raise ValueError("Invalid number of visual stages.")
        self.model.requires_grad_(False)
        for block in list(self.model.blocks.children())[-stages:]:
            block.requires_grad_(True)
        self.model.conv_head.requires_grad_(True)
        self.model.bn2.requires_grad_(True)
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "EmotiEffLib/enet_b2_7", "revision": VISION_REVISION,
            "sha256": self.artifact_sha256, "adapted_sha256": self.adapted_sha256,
            "parameter_count": self.parameter_count(), "feature_dim": self.feature_dim,
            "preprocessing": "RGB; resize 260x260 bilinear PIL; ImageNet normalization; max pooling",
            "diagnostic_labels": list(self.labels),
        }


class TextEncoder(nn.Module):
    """DeBERTa-v3-large with masked-mean features for a MELD-trained head."""

    feature_dim = TEXT_FEATURE_DIM

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        max_length: int = 128,
        batch_size: int = 16,
        mixed_precision: bool = True,
        adapted_state_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= max_length <= 512 or batch_size < 1:
            raise ValueError("max_length must be 1..512 and batch_size must be positive.")
        self.device = _device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local text model directory not found: {self.model_path}")
        source_weights = self.model_path / "pytorch_model.bin"
        if not source_weights.is_file():
            raise FileNotFoundError(f"Pinned text weights not found: {source_weights}")
        self.artifact_sha256 = sha256_file(source_weights)
        if self.artifact_sha256 != TEXT_SHA256:
            raise ValueError("The text checkpoint does not match the pinned Microsoft artifact.")
        # The slow SentencePiece tokenizer avoids on-the-fly tokenizer conversion.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=False, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
        )
        if self.model.config.hidden_size != self.feature_dim:
            raise RuntimeError("Expected DeBERTa-v3-large with 1024 hidden dimensions.")
        if count_parameters(self.model) != TEXT_PARAMETER_COUNT:
            raise RuntimeError("The text parameter inventory does not match DeBERTa-v3-large.")
        if adapted_state_path is not None:
            self.model.load_state_dict(torch.load(adapted_state_path, map_location="cpu", weights_only=True))
        self.adapted_sha256 = sha256_file(adapted_state_path) if adapted_state_path else None
        self.model.requires_grad_(False)
        self.model.to(self.device).eval()

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Text inputs must be nonempty strings.")
        return self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, **tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        # Accumulate in float32 to avoid half-precision overflow in pooling.
        return (hidden.float() * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)

    def forward_tensors(self, **tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(**tokens)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("Pass a list of texts, even for one utterance.")
        if len(texts) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = {key: value.to(self.device) for key, value in self.tokenize(texts[start:start + self.batch_size]).items()}
                with _autocast(self.device, self.mixed_precision):
                    features = self.forward(**tokens)
                outputs.append(features.float().cpu().numpy())
        return _check_features(np.concatenate(outputs), self.feature_dim)

    def parameter_count(self) -> int:
        return count_parameters(self.model)

    def unfreeze_last_layers(self, layers: int = 2) -> int:
        if not 1 <= layers <= len(self.model.encoder.layer):
            raise ValueError("Invalid number of text layers.")
        self.model.requires_grad_(False)
        for layer in self.model.encoder.layer[-layers:]:
            layer.requires_grad_(True)
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "microsoft/deberta-v3-large", "parameter_count": self.parameter_count(),
            "revision": TEXT_REVISION, "sha256": self.artifact_sha256,
            "feature_dim": self.feature_dim, "max_length": self.max_length,
            "pooling": "attention-mask mean, including non-padding special tokens",
            "adapted_sha256": self.adapted_sha256,
        }
