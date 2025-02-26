from __future__ import annotations

from typing import List

from pydantic import BaseModel


class ImageEmbeddingResponse(BaseModel):
    items: List[dict]

    class Config:
        # Ignore extra fields like 'cost' in the input data
        extra = "ignore"
