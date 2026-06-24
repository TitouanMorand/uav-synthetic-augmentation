"""Lazy Diffusers backend for UAV synthetic augmentation."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DiffusionBackendConfig:
    img2img_model_id: str = "runwayml/stable-diffusion-v1-5"
    inpaint_model_id: str = "runwayml/stable-diffusion-inpainting"
    device_preference: str = "auto"


class DiffusionBackend:
    """Small wrapper around pretrained Diffusers pipelines.

    Pipelines are loaded lazily because model download/loading is expensive and
    not needed for CLI help, manifest parsing, or preview generation.
    """

    def __init__(self, config: DiffusionBackendConfig):
        self.config = config
        self.device = self.detect_device(config.device_preference)
        self.dtype = self._select_dtype(self.device)
        self._img2img_pipe = None
        self._inpaint_pipe = None

    @staticmethod
    def detect_device(preference: str = "auto") -> str:
        """Return cuda, mps, or cpu with graceful fallback."""
        import torch

        requested = preference.lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if requested == "cpu":
            return "cpu"
        print(f"Requested device '{preference}' is unavailable; falling back to auto device.")
        return DiffusionBackend.detect_device("auto")

    @staticmethod
    def _select_dtype(device: str):
        import torch

        if device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    @staticmethod
    def _prepare_image(image: Image.Image | np.ndarray) -> tuple[Image.Image, tuple[int, int]]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        pil = image.convert("RGB")
        return pil, pil.size

    @staticmethod
    def _restore_size(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        if image.size == target_size:
            return image
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def _memory_saving_setup(self, pipe):
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def _load_img2img_pipe(self):
        if self._img2img_pipe is not None:
            return self._img2img_pipe
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise ImportError("diffusers is required for diffusion generation. Install requirements.txt.") from exc

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.img2img_model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe = pipe.to(self.device)
        self._memory_saving_setup(pipe)
        self._img2img_pipe = pipe
        return pipe

    def _load_inpaint_pipe(self):
        if self._inpaint_pipe is not None:
            return self._inpaint_pipe
        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:
            raise ImportError("diffusers is required for diffusion generation. Install requirements.txt.") from exc

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.config.inpaint_model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe = pipe.to(self.device)
        self._memory_saving_setup(pipe)
        self._inpaint_pipe = pipe
        return pipe

    def _generator(self, seed: int):
        import torch

        generator_device = self.device if self.device == "cuda" else "cpu"
        return torch.Generator(device=generator_device).manual_seed(seed)

    def generate_img2img(
        self,
        image: Image.Image | np.ndarray,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        """Generate an image-to-image variant and restore original resolution."""
        source, target_size = self._prepare_image(image)
        pipe = self._load_img2img_pipe()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self._generator(seed),
        ).images[0]
        return self._restore_size(result.convert("RGB"), target_size)

    def generate_inpaint(
        self,
        image: Image.Image | np.ndarray,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        """Generate an inpainted variant and restore original resolution.

        Diffusers mask convention: white pixels are generated, black pixels are preserved.
        """
        source, target_size = self._prepare_image(image)
        mask_l = mask.convert("L").resize(target_size, Image.Resampling.NEAREST)
        pipe = self._load_inpaint_pipe()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            mask_image=mask_l,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self._generator(seed),
        ).images[0]
        return self._restore_size(result.convert("RGB"), target_size)
