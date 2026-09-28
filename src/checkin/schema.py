"""Validated state emitted before generation; scores are not calibrated beliefs."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from .settings import LABELS

class Emotion(BaseModel):
    label: Literal["neutral","surprise","fear","sadness","joy","disgust","anger"]
    probabilities: dict[str,float]
    source: Literal["fusion","text_fallback"]
    score_semantics: str = "uncalibrated_softmax"

    @field_validator("probabilities")
    @classmethod
    def check_probabilities(cls, value):
        if set(value) != set(LABELS) or any(not 0 <= p <= 1 for p in value.values()):
            raise ValueError("Expected seven finite MELD probabilities")
        if abs(sum(value.values())-1)>1e-4:
            raise ValueError("Probabilities must sum to one")
        return value

class Response(BaseModel):
    text: str = ""
    status: Literal["pending","streaming","complete","error","cancelled"] = "pending"
    error: str | None = None

class CheckInState(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    turn_id: str
    input: dict
    emotion: Emotion
    vision: dict
    modalities: dict
    modality_disagreement: bool
    timing: dict[str,float|None]
    interaction: dict = Field(default_factory=dict)
    game: dict = Field(default_factory=dict)
    response: Response = Field(default_factory=Response)
