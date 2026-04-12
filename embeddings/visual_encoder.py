# """GME encoder extracted from the MRMR code path, with only the retrieval pieces kept."""

# from __future__ import annotations

# import logging
# import math
# from typing import List, Sequence

# import numpy as np
# import torch
# from PIL import Image
# from tqdm.auto import tqdm

# logger = logging.getLogger(__name__)

# IMAGE_FACTOR = 28
# MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 16384 * 28 * 28
# MAX_RATIO = 200


# def round_by_factor(number: int, factor: int) -> int:
#     return round(number / factor) * factor


# def ceil_by_factor(number: int, factor: int) -> int:
#     return math.ceil(number / factor) * factor


# def floor_by_factor(number: int, factor: int) -> int:
#     return math.floor(number / factor) * factor


# def smart_resize(
#     height: int,
#     width: int,
#     factor: int = IMAGE_FACTOR,
#     min_pixels: int = MIN_PIXELS,
#     max_pixels: int = MAX_PIXELS,
# ) -> tuple[int, int]:
#     h_bar = max(factor, round_by_factor(height, factor))
#     w_bar = max(factor, round_by_factor(width, factor))

#     if h_bar * w_bar > max_pixels:
#         beta = math.sqrt((height * width) / max_pixels)
#         h_bar = floor_by_factor(height / beta, factor)
#         w_bar = floor_by_factor(width / beta, factor)
#     elif h_bar * w_bar < min_pixels:
#         beta = math.sqrt(min_pixels / (height * width))
#         h_bar = ceil_by_factor(height * beta, factor)
#         w_bar = ceil_by_factor(width * beta, factor)

#     if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
#         if h_bar > w_bar:
#             h_bar = w_bar * MAX_RATIO
#         else:
#             w_bar = h_bar * MAX_RATIO

#     return h_bar, w_bar


# def fetch_image(image: str | Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
#     if isinstance(image, Image.Image):
#         image_obj = image
#     else:
#         image_obj = Image.open(image)
#     image_obj = image_obj.convert("RGB")
#     width, height = image_obj.size
#     resized_height, resized_width = smart_resize(
#         height,
#         width,
#         factor=IMAGE_FACTOR,
#         min_pixels=MIN_PIXELS,
#         max_pixels=max_pixels,
#     )
#     return image_obj.resize((resized_width, resized_height))


# class _Encoder(torch.nn.Module):
#     """The exact GME encoder body from the MRMR code path."""

#     def __init__(self, base, processor, max_length: int = 4096, normalize: bool = True):
#         super().__init__()
#         self.base = base
#         self.processor = processor
#         self.max_length = max_length
#         self.normalize = normalize
#         self.processor.tokenizer.padding_side = "right"
#         self.defualt_instruction = "You are a helpful assistant."

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         pixel_values=None,
#         image_grid_thw=None,
#         pooling_mask=None,
#         **kwargs,
#     ):
#         if inputs_embeds is None:
#             inputs_embeds = self.base.model.embed_tokens(input_ids)
#             if pixel_values is not None:
#                 pixel_values = pixel_values.type(self.base.visual.get_dtype())
#                 image_embeds = self.base.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
#                 image_mask = input_ids == self.base.config.image_token_id
#                 inputs_embeds[image_mask] = image_embeds
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(inputs_embeds.device)

#         outputs = self.base.model(
#             input_ids=None,
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#         )

#         pooling_mask = attention_mask if pooling_mask is None else pooling_mask
#         left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]
#         if left_padding:
#             embeddings = outputs.last_hidden_state[:, -1]
#         else:
#             sequence_lengths = pooling_mask.sum(dim=1) - 1
#             batch_size = outputs.last_hidden_state.shape[0]
#             embeddings = outputs.last_hidden_state[
#                 torch.arange(batch_size, device=outputs.last_hidden_state.device),
#                 sequence_lengths,
#             ]
#         if self.normalize:
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
#         return embeddings.contiguous()

#     def embed(self, texts, images, device, instruction=None, **kwargs):
#         instruction = instruction or self.defualt_instruction
#         input_texts, input_images = [], []
#         for text, image in zip(texts, images):
#             input_str = ""
#             if image is None:
#                 input_images = None
#             else:
#                 input_str += "<|vision_start|><|image_pad|><|vision_end|>"
#                 proc_max = getattr(self.processor.image_processor, "max_pixels", MAX_PIXELS)
#                 input_images.append(fetch_image(image, max_pixels=proc_max))
#             if text is not None:
#                 input_str += text
#             msg = (
#                 f"<|im_start|>system\n{instruction}<|im_end|>\n"
#                 f"<|im_start|>user\n{input_str}<|im_end|>\n"
#                 f"<|im_start|>assistant\n<|endoftext|>"
#             )
#             input_texts.append(msg)

#         inputs = self.processor(
#             text=input_texts,
#             images=input_images,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         embeddings = self.forward(**inputs)
#         return embeddings


# class GMEEncoder:
#     def __init__(
#         self,
#         model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
#         device: str = "cuda",
#         max_image_tokens: int = 4096,
#         max_length: int = 4096,
#     ) -> None:
#         self.model_name = model_name
#         self.device = device if torch.cuda.is_available() else "cpu"
#         self.max_image_tokens = max_image_tokens
#         self.max_length = max_length
#         self.model: _Encoder | None = None

#     def load(self) -> None:
#         if self.model is not None:
#             return

#         from transformers import AutoModelForVision2Seq, AutoProcessor

#         torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
#         logger.info("Loading %s on %s...", self.model_name, self.device)
#         base = AutoModelForVision2Seq.from_pretrained(
#             self.model_name,
#             torch_dtype=torch_dtype,
#             low_cpu_mem_usage=True,
#         )
#         processor = AutoProcessor.from_pretrained(
#             self.model_name,
#             min_pixels=MIN_PIXELS,
#             max_pixels=self.max_image_tokens * 28 * 28,
#         )
#         self.model = _Encoder(base, processor, max_length=self.max_length)
#         self.model.eval()
#         self.model.base.to(self.device)
#         logger.info("Model loaded")

#     def unload(self) -> None:
#         if self.model is not None:
#             del self.model
#             self.model = None
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             logger.info("Model unloaded")

#     @staticmethod
#     def _pair_from_item(item: dict) -> tuple[str | None, Image.Image | None]:
#         modality = item.get("modality", "image,text")
#         text = item.get("text")
#         image = item.get("image")

#         if modality == "text":
#             return text, None
#         if modality == "image":
#             return None, image
#         if modality == "image,text":
#             return text, image

#         # Fallback for unexpected modality strings.
#         return text, image

#     def _encode_batch(self, texts: Sequence[str | None], images: Sequence[Image.Image | None], instruction: str | None):
#         assert self.model is not None
#         with torch.inference_mode():
#             embeddings = self.model.embed(
#                 list(texts),
#                 list(images),
#                 device=self.device,
#                 instruction=instruction,
#             )
#         return embeddings.float().cpu().numpy()

#     def _encode_batch_with_backoff(
#         self,
#         texts: Sequence[str | None],
#         images: Sequence[Image.Image | None],
#         instruction: str | None,
#     ) -> np.ndarray:
#         try:
#             return self._encode_batch(texts, images, instruction)
#         except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
#             message = str(exc).lower()
#             if "out of memory" not in message:
#                 raise
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             if len(texts) == 1:
#                 raise RuntimeError(
#                     "OOM while encoding a single item. Reduce --batch_size, reduce --max_image_tokens, "
#                     "or use the 2B GME model for debugging."
#                 ) from exc
#             mid = len(texts) // 2
#             left = self._encode_batch_with_backoff(texts[:mid], images[:mid], instruction)
#             right = self._encode_batch_with_backoff(texts[mid:], images[mid:], instruction)
#             return np.concatenate([left, right], axis=0)

#     def encode_batch_items(self, items: List[dict], instruction: str | None = None):
#         self.load()
#         ids, texts, images = [], [], []
#         for item in items:
#             text, image = self._pair_from_item(item)
#             ids.append(item["id"])
#             texts.append(text)
#             images.append(image)
#         embeddings = self._encode_batch_with_backoff(texts, images, instruction)
#         return ids, embeddings

#     def encode_items(
#         self,
#         items: List[dict],
#         instruction: str | None = None,
#         batch_size: int = 4,
#         show_progress: bool = True,
#         desc: str = "Encoding",
#     ):
#         self.load()
#         all_ids: list[str] = []
#         all_embeddings: list[np.ndarray] = []

#         pending: list[dict] = []
#         pending_has_image: bool | None = None

#         iterator = tqdm(items, disable=not show_progress, desc=desc)
#         for item in iterator:
#             _, image = self._pair_from_item(item)
#             has_image = image is not None
#             if pending_has_image is None:
#                 pending_has_image = has_image
#             if pending and (has_image != pending_has_image or len(pending) >= batch_size):
#                 batch_ids, batch_embeddings = self.encode_batch_items(pending, instruction=instruction)
#                 all_ids.extend(batch_ids)
#                 all_embeddings.extend(batch_embeddings)
#                 pending = []
#                 pending_has_image = has_image
#             pending.append(item)

#         if pending:
#             batch_ids, batch_embeddings = self.encode_batch_items(pending, instruction=instruction)
#             all_ids.extend(batch_ids)
#             all_embeddings.extend(batch_embeddings)

#         return all_ids, np.asarray(all_embeddings)


# def create_encoder(
#     model_name: str,
#     device: str = "cuda",
#     max_image_tokens: int = 4096,
#     max_length: int = 4096,
# ) -> GMEEncoder:
#     return GMEEncoder(
#         model_name=model_name,
#         device=device,
#         max_image_tokens=max_image_tokens,
#         max_length=max_length,
#     )
"""GME encoder using the official get_fused_embeddings API.

This avoids the shape-mismatch bug between tokenizer and visual encoder
that occurs with the manual chat-template approach.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class GMEEncoder:
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_image_tokens: int = 4096,
        max_length: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_image_tokens = max_image_tokens
        self.max_length = max_length
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModel

        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        logger.info("Loading %s on %s...", self.model_name, self.device)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
        )
        self._model.eval()
        logger.info("Model loaded")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    # ------------------------------------------------------------------
    # Extract (text, image) from an item dict based on modality
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_from_item(item: dict) -> tuple[str | None, Image.Image | None]:
        modality = item.get("modality", "image,text")
        text = item.get("text")
        image = item.get("image")

        if modality == "text":
            return text, None
        if modality == "image":
            return None, image
        if modality == "image,text":
            return text, image

        return text, image

    # ------------------------------------------------------------------
    # Core encoding via official GME API
    # ------------------------------------------------------------------

    def _encode_batch(
        self,
        texts: Sequence[str | None],
        images: Sequence[Image.Image | None],
        instruction: str | None,
    ) -> np.ndarray:
        """Encode a batch using the official GME API. No shape mismatch possible."""
        assert self._model is not None

        has_any_image = any(img is not None for img in images)
        has_any_text = any(t is not None for t in texts)

        with torch.inference_mode():
            if has_any_image and has_any_text:
                emb = self._model.get_fused_embeddings(
                    texts=list(texts),
                    images=list(images),
                    instruction=instruction,
                )
            elif has_any_image:
                emb = self._model.get_image_embeddings(
                    images=list(images),
                )
            else:
                emb = self._model.get_text_embeddings(
                    texts=list(texts),
                    instruction=instruction,
                )

        return emb.float().cpu().numpy()

    def _encode_batch_with_backoff(
        self,
        texts: Sequence[str | None],
        images: Sequence[Image.Image | None],
        instruction: str | None,
    ) -> np.ndarray:
        try:
            return self._encode_batch(texts, images, instruction)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            if "out of memory" not in message:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(texts) == 1:
                # Single item OOM with image → retry text-only
                if any(img is not None for img in images):
                    logger.warning("  OOM on single item, retrying text-only")
                    return self._encode_batch(texts, [None], instruction)
                raise RuntimeError(
                    "OOM while encoding a single text-only item. "
                    "Reduce --batch_size or --max_length."
                ) from exc
            mid = len(texts) // 2
            left = self._encode_batch_with_backoff(texts[:mid], images[:mid], instruction)
            right = self._encode_batch_with_backoff(texts[mid:], images[mid:], instruction)
            return np.concatenate([left, right], axis=0)

    # ------------------------------------------------------------------
    # High-level encode interfaces (same as before, unchanged)
    # ------------------------------------------------------------------

    def encode_batch_items(self, items: List[dict], instruction: str | None = None):
        self.load()
        ids, texts, images = [], [], []
        for item in items:
            text, image = self._pair_from_item(item)
            ids.append(item["id"])
            texts.append(text)
            images.append(image)
        embeddings = self._encode_batch_with_backoff(texts, images, instruction)
        return ids, embeddings

    def encode_items(
        self,
        items: List[dict],
        instruction: str | None = None,
        batch_size: int = 4,
        show_progress: bool = True,
        desc: str = "Encoding",
    ):
        self.load()
        all_ids: list[str] = []
        all_embeddings: list[np.ndarray] = []

        pending: list[dict] = []
        pending_has_image: bool | None = None

        iterator = tqdm(items, disable=not show_progress, desc=desc)
        for item in iterator:
            _, image = self._pair_from_item(item)
            has_image = image is not None
            if pending_has_image is None:
                pending_has_image = has_image
            if pending and (has_image != pending_has_image or len(pending) >= batch_size):
                batch_ids, batch_embeddings = self.encode_batch_items(pending, instruction=instruction)
                all_ids.extend(batch_ids)
                all_embeddings.extend(batch_embeddings)
                pending = []
                pending_has_image = has_image
            pending.append(item)

        if pending:
            batch_ids, batch_embeddings = self.encode_batch_items(pending, instruction=instruction)
            all_ids.extend(batch_ids)
            all_embeddings.extend(batch_embeddings)

        return all_ids, np.asarray(all_embeddings)


def create_encoder(
    model_name: str,
    device: str = "cuda",
    max_image_tokens: int = 4096,
    max_length: int = 4096,
) -> GMEEncoder:
    return GMEEncoder(
        model_name=model_name,
        device=device,
        max_image_tokens=max_image_tokens,
        max_length=max_length,
    )
