import re
import types
import io
import torch
from PIL import Image
from qwen_vl_utils import fetch_image

from transformers import (
    ProcessorMixin, 
    SiglipImageProcessor, 
    BatchFeature, 
    Qwen2VLImageProcessor,
    PreTrainedTokenizer
)

from .utils import (
    process_anyres_image,
    BLACK_IMG_ENV, 
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VI_END_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    SEQ_MAX_LEN,  
)

siglip_processor_config = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_processor_type": "SiglipImageProcessor",
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "processor_class": "SiglipProcessor",
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {
        "height": 384,
        "width": 384
    }
}

qwen2vl_processor_config = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "image_processor_type": "Qwen2VLImageProcessor",
    "processor_class": "Qwen2VLProcessor"
}

class ValleyProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    optional_attributes = [
        "max_pixels", 
        "min_pixels", 
        "anyres", 
        "only_crop_single_image", 
        "grid_pinpoints", 
        "use_special_start_end_token",
        "only_navit",
        "chat_template",
    ]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)
        self.black_img = BLACK_IMG_ENV
        self.siglip_image_processor = SiglipImageProcessor.from_dict(siglip_processor_config)
        self.qwen2vl_image_processor = Qwen2VLImageProcessor.from_dict(
            qwen2vl_processor_config, 
        )
        
        self.anyres = kwargs.get("anyres", True)
        self.grid_pinpoints = kwargs.get("grid_pinpoints", "(1x1),...,(3x3)")
        self.only_crop_single_image = kwargs.get("only_crop_single_image", True)
        self.use_special_start_end_token = kwargs.get("use_special_start_end_token", True)
        self.only_navit = kwargs.get("only_navit", False)

    def preprocess_images_siglip(self, images) -> torch.FloatTensor:
        if isinstance(images[0], str):
            images_pil = [Image.open(img).convert("RGB") for img in images]
        elif isinstance(images[0], Image.Image):
            images_pil = [img.convert("RGB") for img in images]
        elif isinstance(images[0], bytes):
            images_pil = [Image.open(io.BytesIO(img)).convert("RGB") for img in images]
        else:
            raise ValueError("unsupported type")

        processed_images = []
        have_multi_images = len(images_pil) > 1
        for img in images_pil:
            if self.anyres:
                if not self.only_crop_single_image or not have_multi_images:
                    image = process_anyres_image(img, self.siglip_image_processor, self.grid_pinpoints)
                else:
                    image = [self.siglip_image_processor(img, return_tensors="pt")["pixel_values"][0]]
            else:
                image = self.siglip_image_processor(img, return_tensors="pt")["pixel_values"][0]
            
            processed_images.append(image)

        if not self.anyres:
            return torch.stack(processed_images, dim=0)
        else:
            return [torch.stack(img, dim=0) for img in processed_images]
    
    def preprocess_images_qwen2vl(self, images) -> dict:
        if isinstance(images[0], str):
            images_pil = [Image.open(img).convert("RGB") for img in images]
        elif isinstance(images[0], Image.Image):
            images_pil = [img.convert("RGB") for img in images]
        elif isinstance(images[0], bytes):
            images_pil = [Image.open(io.BytesIO(img)).convert("RGB") for img in images]
        else:
            raise ValueError("unsupported type")

        image_sizes = [[x.size for x in images_pil]]
        data_dict_qwen2vl = self.qwen2vl_image_processor(
            [fetch_image({"image": img}) for img in images_pil], 
            return_tensors="pt"
        )

        data_dict_qwen2vl["image_sizes"] = image_sizes

        return data_dict_qwen2vl

    def preprocess_multimodal(self, conversations):
        for sentence in conversations:
            if sentence["role"] == "system":
                continue
            segs = re.split(DEFAULT_IMAGE_TOKEN, sentence["content"])
            if self.use_special_start_end_token:
                sentence["content"] = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN).join(segs)
            else:
                sentence["content"] = DEFAULT_IMAGE_TOKEN.join(segs)

        return conversations

    def preprocess_qwen2(
        self,
        conversations,
        tokenizer: PreTrainedTokenizer,
        has_image: bool = False,
        inference: bool = False,
        only_mask_system: bool = False,
    ) -> dict:
        conv = types.SimpleNamespace(
            system="You are a helpful assistant.",
            roles=("user", "assistant"),
            version="qwen2",
            offset=0,
            sep="<|im_start|>",
            sep2="<|im_end|>\n",
        )

        # Check system prompt
        assert conversations[0]["role"] == "system"
        if conversations[0]["content"] == None:
            conversations[0]["content"] = conv.system # use default system prompt
        
        # Check conversation sequence
        for j, sentence in enumerate(conversations[1:]):
            role = sentence["role"]
            assert role == conv.roles[j % 2], "The conversation sequence is incorrect."
        
        conversation_str = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=inference)
        
        # Mask targets
        rounds = conversation_str.split(conv.sep2)
        input_ids_ = torch.tensor([], dtype=torch.int64)
        targets_ = torch.tensor([], dtype=torch.int64)
        for i, rou in enumerate(rounds):
            if rou == "":
                continue
            if (not inference) or (i < (len(rounds) - 1)):
                rou += conv.sep2
            if has_image:
                cur_input_ids_ = self.tokenizer_image_token(rou, tokenizer, return_tensors='pt')
                input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
                if only_mask_system:
                    mask_len = len(self.tokenizer_image_token(re.sub(rf'{conv.roles[0]}\n[\s\S]*', f'{conv.roles[0]}:', rou),
                                                        tokenizer))
                else:
                    mask_len = len(self.tokenizer_image_token(re.sub(rf'{conv.roles[1]}\n[\s\S]*', f'{conv.roles[1]}:', rou),
                                                        tokenizer))
                targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
            else:
                cur_input_ids_ = tokenizer(rou, return_tensors='pt')["input_ids"][0, :]
                input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
                mask_len = len(tokenizer(re.sub(rf'{conv.roles[1]}\n[\s\S]*', rf'{conv.roles[1]}:', rou))["input_ids"][:])
                targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
        
        return {"input_ids": input_ids_, "labels": targets_}


    def tokenizer_image_token(
        self,
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors=None,
    ):
        def split_with_token(string, token):
            result = string.split(token)
            for i in range(len(result) - 1):
                result.insert(i * 2 + 1, token)
            return result

        if len(prompt) > SEQ_MAX_LEN:
            raise ValueError("sequence is too long !!!")

        prompt_chunks = split_with_token(prompt, DEFAULT_IMAGE_TOKEN)
        input_ids, offset = ([tokenizer.bos_token_id], 1) if getattr(tokenizer,'bos_token',None) else ([], 0)
        token2index = {DEFAULT_IMAGE_TOKEN: image_token_index}
        for chunk in prompt_chunks:
            if chunk in token2index:
                input_ids.append(token2index[chunk])
            else:
                chunk_ids = tokenizer(chunk).input_ids
                if chunk_ids[0] != getattr(tokenizer,'bos_token_id', None):
                    offset = 0
                input_ids.extend(chunk_ids[offset:])

        if return_tensors is not None:
            if return_tensors == "pt":
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        return input_ids


    def __call__(self, messages, inference=True, **kwargs) -> BatchFeature:
        max_pixels=kwargs.get("max_pixels", self.max_pixels)
        min_pixels=kwargs.get("min_pixels", self.min_pixels)
        if max_pixels is not None:
            self.qwen2vl_image_processor.max_pixels = max_pixels
        if min_pixels is not None:
            self.qwen2vl_image_processor.min_pixels = min_pixels

        # Deal with images
        if "images" not in messages or not messages["images"] or not messages["images"][0]:
            images = [self.black_img]
        elif type(messages["images"]) == str:
            images = [messages["images"]]
        else:
            images = messages["images"]

        # Deal with conversations
        conversations = messages["conversations"]
        if conversations[0]["role"] != "system":
            conversations = [{"role":"system", "content": None}] + conversations  # dummy system prompt
        
        # Insert special token `<image>`
        assert conversations[1]["role"] == "user"
        if images and "<image>" not in conversations[1]["content"]:
            image_token = " ".join(["<image>"] * len(images))
            conversations[1]["content"] = f"{image_token}\n{conversations[1]['content']}"
        
        # The last message should be assistant if inference=True
        if inference:
            assert conversations[-1]["role"] == "user", "the last message should be assistant if inference=True"
        
        # Image preprocess
        if self.only_navit:
            precessed_images_siglip = None
        else:
            precessed_images_siglip = self.preprocess_images_siglip(images)
        processed_data_dict_qwen2vl = self.preprocess_images_qwen2vl(images)
        source = self.preprocess_multimodal(conversations)
        data_dict = self.preprocess_qwen2(source, self.tokenizer, has_image=True, only_mask_system=False, inference=inference)
        
        # Construct batch data
        data_dict["input_ids"] = data_dict["input_ids"].unsqueeze(0) # batch_size = 1
        data_dict["labels"] = data_dict["labels"].unsqueeze(0)
        data_dict["images"] = [precessed_images_siglip]
        
        return BatchFeature(data={**data_dict, **processed_data_dict_qwen2vl})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)


    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)