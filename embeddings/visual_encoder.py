"""
Multimodal encoder: Qwen3-VL-Embedding-2B

Encodes (text, image) pairs into dense vectors.
Pooling: EOS token's last hidden state, L2 normalized.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MultimodalEncoder:
    """
    Qwen3-VL-Embedding-2B encoder.

    Usage:
        encoder = MultimodalEncoder()
        vec = encoder.encode(text="What is this?", image=pil_img)
        # vec: numpy array, shape (2048,), L2 normalized
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.device = self.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.model_name}...")

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model.eval()
        logger.info(f"Model loaded on {self.device}")

    def unload(self):
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded")

    @torch.no_grad()
    def encode(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        instruction: str = "",
        max_image_pixels: int = 512 * 512,
    ) -> np.ndarray:
        """
        Encode a single (text, image) pair.

        Returns:
            embedding: shape (hidden_dim,), L2 normalized
        """
        self.load()

        # 限制图片大小，防止 OOM
        if image is not None:
            w, h = image.size
            pixels = w * h
            if pixels > max_image_pixels:
                scale = (max_image_pixels / pixels) ** 0.5
                new_w = max(int(w * scale), 28)
                new_h = max(int(h * scale), 28)
                image = image.resize((new_w, new_h), Image.LANCZOS)

        # 尝试带图编码，OOM 则回退到纯文本
        try:
            return self._encode_inner(text, image, instruction)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.warning(f"  OOM with image, retrying text-only")
                return self._encode_inner(text, None, instruction)
            raise

    def _encode_inner(self, text, image, instruction):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})

        full_text = text
        if instruction:
            full_text = f"Instruct: {instruction}\nQuery: {text}"
        content.append({"type": "text", "text": full_text})

        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        outputs = self._model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # EOS token pooling
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden[0, -1, :].float().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_batch(
        self,
        items: Dict[str, dict],
        cache_path: Optional[Path] = None,
        item_type: str = "item",
        use_instruction: bool = False,
    ):
        """
        Encode a dict of items. Each item has "text" and optionally "image", "instruction".
        Results are cached to .npz.

        Args:
            items: {id: {"text": str, "image": PIL.Image or None, ...}}
            cache_path: path to save/load cache
            item_type: for logging ("query" or "corpus")
            use_instruction: whether to prepend instruction

        Returns:
            ids: List[str]
            embeddings: np.ndarray of shape (N, hidden_dim)
        """
        # Try loading cache
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached {item_type} embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["ids"].tolist(), data["embeddings"]

        self.load()

        logger.info(f"Encoding {len(items)} {item_type}s...")
        ids = []
        embeddings = []

        t0 = time.time()
        for i, (item_id, item) in enumerate(items.items()):
            instruction = ""
            if use_instruction and "instruction" in item:
                instruction = item["instruction"]

            emb = self.encode(
                text=item["text"],
                image=item.get("image"),
                instruction=instruction,
            )

            ids.append(item_id)
            embeddings.append(emb)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                remaining = (len(items) - i - 1) / speed
                logger.info(f"  [{item_type}] {i+1}/{len(items)} "
                            f"({speed:.1f}/s, ~{remaining/60:.1f}min left)")

        embeddings = np.stack(embeddings)

        # Save cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, ids=np.array(ids), embeddings=embeddings)
            logger.info(f"  Cached to {cache_path}")

        elapsed = time.time() - t0
        logger.info(f"  Encoded {len(items)} {item_type}s in {elapsed/60:.1f}min")

        return ids, embeddings


class GMEEncoder:
    """
    GME-Qwen2-VL encoder (2B or 7B).
    Requires transformers<4.52.0.

    Usage:
        encoder = GMEEncoder(model_name="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")
        vec = encoder.encode(text="What is this?", image=pil_img)
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self):
        if self._model is not None:
            return

        from transformers import AutoModel

        self.device = self.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.model_name}...")

        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        logger.info(f"Model loaded on {self.device}")

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded")

    @torch.no_grad()
    def encode(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        instruction: str = "",
        max_image_pixels: int = 512 * 512,
    ) -> np.ndarray:
        self.load()

        # 限制图片大小
        if image is not None:
            w, h = image.size
            if w * h > max_image_pixels:
                scale = (max_image_pixels / (w * h)) ** 0.5
                new_w = max(int(w * scale), 28)
                new_h = max(int(h * scale), 28)
                image = image.resize((new_w, new_h), Image.LANCZOS)

        try:
            return self._encode_inner(text, image, instruction)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.warning(f"  OOM with image, retrying text-only")
                return self._encode_inner(text, None, instruction)
            raise

    def _encode_inner(self, text, image, instruction):
        instr = instruction if instruction else "Represent the user's input."

        if image is not None and text:
            # Fused: text + image
            emb = self._model.get_fused_embeddings(
                texts=[text],
                images=[image],
                instruction=instr,
                is_query=bool(instruction),
            )
        elif image is not None:
            # Image only
            emb = self._model.get_image_embeddings(
                images=[image],
                is_query=bool(instruction),
            )
        else:
            # Text only
            emb = self._model.get_text_embeddings(
                texts=[text],
                instruction=instr,
                is_query=bool(instruction),
            )

        # emb is (1, dim) tensor, already L2 normalized by GME
        return emb[0].float().cpu().numpy()

    def encode_batch(
        self,
        items: Dict[str, dict],
        cache_path: Optional[Path] = None,
        item_type: str = "item",
        use_instruction: bool = False,
    ):
        """Same interface as MultimodalEncoder.encode_batch."""
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached {item_type} embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["ids"].tolist(), data["embeddings"]

        self.load()

        logger.info(f"Encoding {len(items)} {item_type}s...")
        ids = []
        embeddings = []

        t0 = time.time()
        for i, (item_id, item) in enumerate(items.items()):
            instruction = ""
            if use_instruction and "instruction" in item:
                instruction = item["instruction"]

            emb = self.encode(
                text=item["text"],
                image=item.get("image"),
                instruction=instruction,
            )

            ids.append(item_id)
            embeddings.append(emb)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                remaining = (len(items) - i - 1) / speed
                logger.info(f"  [{item_type}] {i+1}/{len(items)} "
                            f"({speed:.1f}/s, ~{remaining/60:.1f}min left)")

        embeddings = np.stack(embeddings)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, ids=np.array(ids), embeddings=embeddings)
            logger.info(f"  Cached to {cache_path}")

        elapsed = time.time() - t0
        logger.info(f"  Encoded {len(items)} {item_type}s in {elapsed/60:.1f}min")

        return ids, embeddings


def create_encoder(model_name: str, device: str = "cuda"):
    """
    Factory: pick the right encoder class based on model name.

    Usage:
        encoder = create_encoder("Qwen/Qwen3-VL-Embedding-2B")
        encoder = create_encoder("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")
    """
    name_lower = model_name.lower()
    if "gme" in name_lower:
        return GMEEncoder(model_name=model_name, device=device)
    else:
        return MultimodalEncoder(model_name=model_name, device=device)