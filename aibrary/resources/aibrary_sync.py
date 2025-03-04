import os
from typing import List, Optional

import httpx
import openai

import aibrary as aib
from aibrary.resources.image_embedding import ImageEmbedding
from aibrary.resources.models import Model
from aibrary.resources.object_detection import ObjectDetectionClient
from aibrary.resources.ocr import OCRClient
from aibrary.resources.translation import TranslationClient


class AiBrary(openai.OpenAI):
    """Base class for OpenAI modules."""

    def __init__(
        self,
        *,
        api_key: str = None,
        **kwargs,
    ):
        """Initialize with API key and base URL."""
        if api_key is None:
            api_key = os.environ.get("AIBRARY_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the AIBRARY_API_KEY environment variable"
            )
        self.api_key = api_key
        openai.api_key = api_key

        self.base_url = aib.base_url
        openai.base_url = self.base_url
        super().__init__(api_key=self.api_key, base_url=self.base_url, **kwargs)

        """Initialize all modules."""
        self.translation = TranslationClient(
            api_key=api_key, base_url=self.base_url
        ).automatic_translation
        self.ocr = OCRClient(base_url=self.base_url, api_key=self.api_key).process_ocr
        self.object_detection = ObjectDetectionClient(
            base_url=self.base_url, api_key=self.api_key
        ).process_object_detection
        self.image_embedding = ImageEmbedding(
            base_url=self.base_url, api_key=self.api_key
        ).process_image_embedding

    def get_all_models(
        self, return_as_objects: bool = True, filter_category: Optional[str] = None
    ) -> Optional[List[Model]]:
        """
        Fetches all models from the given base URL using the provided API token.

        Args:
            return_as_objects (bool): If True, converts JSON data to Model objects.
            filter_category (str): Filter models based on given category.

        Returns:
            Optional[List[Model]]: A list of Model objects or raw JSON data.

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}dashboard/models",
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()["models"]
        if filter_category:
            data = [
                item
                for item in data
                if item.get("category").lower() == filter_category.lower()
            ]
        if return_as_objects:
            return [Model(**item) for item in data]
        return data
