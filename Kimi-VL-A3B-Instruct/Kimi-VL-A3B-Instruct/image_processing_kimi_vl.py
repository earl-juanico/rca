"""Image processor class for KimiVL."""

import math
import numpy as np
from PIL import Image
from typing import Optional, Union

import torch
from torchvision.transforms import functional as TF
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class KimiVLImageProcessor(BaseImageProcessor):
    model_type = "kimi_vl"

    def __init__(
        self,
        patch_size: int = 14,
        pad_input: bool = False,
        image_mean: tuple[float, float, float] = OPENAI_DATASET_MEAN,
        image_std: tuple[float, float, float] = OPENAI_DATASET_STD,
        in_token_limit: int = 4096,
        merge_kernel_size: list[int, int] = [2, 2],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.pad_input = pad_input
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = merge_kernel_size

    def rescale(
        self, image: Image.Image, merge_kernel_size: list[int, int] = [2, 2]
    ) -> Image.Image:
        w, h = image.size
        patch_size = self.patch_size

        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // patch_size) * (h // patch_size)))
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        if self.pad_input:
            new_w, new_h = image.size
            pad_size_h = merge_kernel_size[0] * patch_size
            pad_size_w = merge_kernel_size[1] * patch_size

            pad_h = (pad_size_h - new_h % pad_size_h) % pad_size_h
            pad_w = (pad_size_w - new_w % pad_size_w) % pad_size_w

            image = TF.pad(image, (0, 0, pad_w, pad_h))
        else:
            new_w, new_h = image.size
            new_w = new_w - new_w % patch_size
            new_h = new_h - new_h % patch_size
            image = TF.center_crop(image, (new_h, new_w))

        w, h = image.size
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        return image

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(image.convert("RGB"))

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return TF.normalize(image, self.image_mean, self.image_std)

    def patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, list[int, int]]:
        patch_size = self.patch_size
        C, H, W = image.shape
        patches = image.reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _preprocess(self, image: ImageInput) -> tuple[torch.Tensor, list[int, int]]:
        """
        Preprocess image and patchify it.

        Args:
            image (`ImageInput`):
                Image to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.

        Returns:
            patches: torch.Tensor
            grid_hw: list[int, int]
        """
        image = self.rescale(image, self.merge_kernel_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        patches, grid_hw = self.patchify(image)
        return patches, grid_hw

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        pixel_values, image_grid_hws = [], []
        for image in images:
            patches, image_grid_hw = self._preprocess(image)
            pixel_values.append(patches)
            image_grid_hws.append(image_grid_hw)
        pixel_values = torch.concat(pixel_values, dim=0)
        image_grid_hws = np.array(image_grid_hws)
        data = {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}

        return BatchFeature(data=data, tensor_type=return_tensors)
