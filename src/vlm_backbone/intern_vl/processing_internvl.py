# processing_internvl.py
from typing import List, Optional, Union
from transformers import ProcessorMixin, BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput, PaddingStrategy, TruncationStrategy
import torch
import re
import numpy as np

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

class InternVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, num_img_tokens=256):
        super().__init__(image_processor, tokenizer)
        self.num_img_tokens = num_img_tokens
        self.img_context_token = "<IMG_CONTEXT>"
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        special_tokens = [self.img_context_token]
        num_added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Union[ImageInput, List[ImageInput]] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ) -> BatchFeature:
        if isinstance(text, str):
            text = [text]
        
        if not isinstance(images, list): 
            images = [images] if images else []
        
        image_flags = [1] if len(images) else [0]
        
        pixel_values = []
        if any(image_flags):
            pixel_values = self.image_processor(
                [img for img in images if img],
                return_tensors=return_tensors
            ).pixel_values
        
        processed_texts = [
            self._insert_image_tokens(t, count) 
            for t, count in zip(text, image_flags)
        ]
        text_inputs = self.tokenizer(
            processed_texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            add_special_tokens=True
        )
        
        return BatchFeature({
            **text_inputs,
            "pixel_values": pixel_values,
            "image_flags": torch.tensor(image_flags),
        }, tensor_type=return_tensors)

    def _insert_image_tokens(self, text: str, image_count: int) -> str:
        if image_count == 0:
            return text
        
        image_tokens = f"<image>{self.img_context_token * self.num_img_tokens * image_count}</image>"
        return text.replace("<image>", image_tokens, 1)