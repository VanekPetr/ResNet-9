from typing import List
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(default="", title="Label of the classification")
    certainty: float = Field(default=0.0, title="Certainty of the classification")


class Classification(BaseModel):
    label: str = Field(default="", title="Label of the classification")
    certainty: float = Field(default=0.0, title="Certainty of the classification")
    predictions: List[Prediction] = Field(default=[], title="All possible labels with their certainty")
