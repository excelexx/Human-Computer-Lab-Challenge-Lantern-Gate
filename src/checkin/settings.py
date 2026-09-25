"""Portable paths and pinned model provenance."""
import os
from pathlib import Path

LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
LABEL_TO_ID = {v: k for k, v in enumerate(LABELS)}
TEXT_MODEL = "microsoft/deberta-v3-large"
TEXT_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
VISION_SHA256 = "e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f"
GGUF_NAME = "Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf"
GGUF_URL = "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/ae44f08e1392f39c0e474af10c3ff8355c8b6688/" + GGUF_NAME
GGUF_SHA256 = "66713ce35a58a82fe87642d4ec13425bf9b9a46800fff5c49a665ef5701439dc"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
VISION_URL = "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/enet_b2_7.pt"
MELD_URL = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
MELD_SHA256 = "a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e"

def runtime_home(value=None):
    home = Path(value or os.environ.get("CHECKIN_HOME") or Path(__file__).resolve().parents[2] / ".artifacts").resolve()
    for part in ["downloads", "models", "manifests", "cache", "checkpoints", "reports", "logs", "vendor"]:
        (home / part).mkdir(parents=True, exist_ok=True)
    return home
