#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from .modeling_vision_tower import build_vision_tower
from .modeling_projector import build_vision_projector
from .utils import get_anyres_image_grid_shape, unpad_image, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from dataclasses import dataclass

@dataclass
class CausalLMOutputWithPastAndMask(CausalLMOutputWithPast):
    attention_mask: Optional[torch.Tensor] = None

class ValleyConfig(Qwen2Config):
    model_type = "valley"

class ValleyMetaModel:
    def __init__(self, config):
        super(ValleyMetaModel, self).__init__(config)
        # Build vision tower
        if hasattr(config, "mm_vision_tower"):
            if getattr(config, "eagle_vision_tower", None) is not None:
                self.vision_tower, self.qwen2vl_vision_tower = build_vision_tower(config, delay_load=False)
            else:
                self.vision_tower = build_vision_tower(config, delay_load=False)
        # Build Projector
        if hasattr(config, "mm_projector_type") and not getattr(config, "only_navit", False):
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if getattr(self.config, "eagle_vision_tower", None) is not None:
            qwen2vl_vision_tower = getattr(self, "qwen2vl_vision_tower", None)
            return vision_tower, qwen2vl_vision_tower
        else:
            return vision_tower

class ValleyMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def split_by_instance(self, original_list, split_sizes):
        start = 0
        sub_lists = []
        for size in split_sizes:
            end = start + size
            sub_list = original_list[start:end]
            sub_lists.append([x.to(self.device) for x in sub_list])
            start = end
        return sub_lists
    
    def encode_images_qwen2vl(self, pixel_values = None, grid_thw = None, split_sizes=None):
        _, qwen2vl_vision_tower = self.get_model().get_vision_tower()
        qwen2vl_image_features = qwen2vl_vision_tower(pixel_values, grid_thw)
        qwen2vl_image_split_sizes = torch.prod(grid_thw[:, 1:3]//2, dim=1)
        qwen2vl_image_features = torch.split(qwen2vl_image_features, qwen2vl_image_split_sizes.tolist(), dim=0)
        qwen2vl_image_features = self.split_by_instance(qwen2vl_image_features, split_sizes)
        return qwen2vl_image_features

    def encode_images(self, images = None, split_sizes = None):
        """
        images: (if not anyres) images.shape = [n,3,336,336] , n = number of images + (number of video) * 8
        images: (if anyres) images.shape = [n,3,336,336] , n = number of tiles * number of images
        """
        if getattr(self.config, "eagle_vision_tower", None) is not None:
            siglip_vision_tower, _ = self.get_model().get_vision_tower()
            image_features = siglip_vision_tower(images)
            image_features = self.get_model().mm_projector(image_features)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)

        if getattr(self.config,'anyres', False) and getattr(self.config, 'max_vision_token', None) is not None:
            assert split_sizes is not None
            image_features = list(torch.split(image_features, split_sizes, dim=0))
            for i, image_feature in enumerate(image_features):
                hidden_dim = image_feature.shape[-1]
                image_tokens = image_feature.shape[0]*image_feature.shape[1]
                if getattr(self.config, "eagle_vision_tower", None) is not None:
                    pass # the max_vision_token will be processed in the unpad image token part
                else:
                    if image_tokens > self.config.max_vision_token:
                        intput_shape = int((image_feature.shape[1])**0.5)
                        output_shape = int((self.config.max_vision_token/image_feature.shape[0])**0.5)
                        image_feature = image_feature.view(image_feature.shape[0],intput_shape, intput_shape, -1).permute(0,3,1,2)
                        m = nn.AdaptiveAvgPool2d(output_shape) # different from roi pooling, but in square image, it seems the same
                        pooling_feature = m(image_feature).permute(0,2,3,1)
                        image_features[i] = pooling_feature.view(image_feature.shape[0], -1, hidden_dim)
                split_sizes = None # have already split, set the flag 

        if getattr(self.config, 'mm_use_im_start_end', False):
            raise ValueError('mm_use_im_start is not support')
        if split_sizes is not None:
            image_features = torch.split(image_features, split_sizes, dim=0)
        
        return image_features

    def get_padding_method(self):
        right_padding = getattr(self, 'right_padding', None)
        # if right_padding flag is setted, ignore training flag. 
        if right_padding is not None:
            method = 'right' if right_padding else 'left'
        # in the other way, use training flag to determine the padding method.
        method = 'right' if self.training else 'left'

        return method

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images,
        image_sizes, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw):

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Step1: Get image embedings
        if type(images) is list or images.ndim == 5:
            # Without slicing the image
            if not getattr(self.config,'anyres', False):
                concat_images = torch.cat([image for image in images], dim=0) # to do batch compute
                split_sizes = [image.shape[0] for image in images]
                
                # Get vision tower feature, check whether only use navit firstly
                if getattr(self.config, 'eagle_vision_tower', None) is not None and getattr(self.config, 'only_navit', False):
                    image_features = None
                else:
                    image_features = self.encode_images(concat_images, split_sizes)
                    image_features = [x.to(self.device) for x in image_features]
                
                # Get Eagle features
                if getattr(self.config, 'eagle_vision_tower', None) is not None:
                    if pixel_values is not None:
                        qwen2vl_image_features = self.encode_images_qwen2vl(pixel_values, image_grid_thw, split_sizes)
                    elif pixel_values_videos is not None:
                        qwen2vl_image_features = self.encode_images_qwen2vl(pixel_values_videos, video_grid_thw, split_sizes)
                    else:
                        qwen2vl_image_features = None

            # Slicing the image, each image contains some sub_images:
            # images = [
            #   [image1_tiles(n1,3,336,336), image2_tiles(n2,3,336,336), ...],
            #   [image1_tiles(n1,3,336,336), image2_tiles(n2,3,336,336), ...], ...
            # ]
            else:
                split_sizes = [len(image) for image in images]
                # Get Eagle features
                if getattr(self.config, "eagle_vision_tower", None) is not None:
                    if pixel_values is not None:
                        qwen2vl_image_features = self.encode_images_qwen2vl(pixel_values, image_grid_thw, split_sizes)
                    elif pixel_values_videos is not None:
                        qwen2vl_image_features = self.encode_images_qwen2vl(pixel_values_videos, video_grid_thw, split_sizes)
                    else:
                        qwen2vl_image_features = None
                
                # Get vision tower feature, check whether only use navit firstly
                if getattr(self.config, 'eagle_vision_tower', None) is not None and getattr(self.config, 'only_navit', False):
                    image_features = None
                else:
                    image_features = []
                    all_concat_images = []
                    all_split_sizes = []
                    for batch_images in images:
                        concat_images = torch.cat([image for image in batch_images], dim=0) # to do batch compute
                        split_sizes = [image.shape[0] for image in batch_images] 
                        all_concat_images.append(concat_images)
                        all_split_sizes.append(split_sizes)
                    all_image_features = self.encode_images(images=torch.cat(all_concat_images, dim=0), split_sizes=sum(all_split_sizes, []))

                    idx = 0
                    for split_sizes in all_split_sizes:
                        batch_image_features = all_image_features[idx:idx+len(split_sizes)]
                        idx += len(split_sizes)
                        if type(batch_image_features[0]) is list:
                            batch_image_features = [torch.cat(x).to(self.device) for x in batch_image_features]
                        else:
                            batch_image_features = [x.view(-1,x.shape[-1]).to(self.device) for x in batch_image_features] # tiles feature need to flatten in token dimention, [n_tiles, T, d] -> [n_tiles * T, d]
                        image_features.append(batch_image_features)

                if getattr(self.config, "eagle_vision_tower", None) is not None and getattr(self.config, 'only_navit', False) == False:
                    # unpad image tokens
                    height = width = self.config.num_patches_per_side
                    new_image_features = []
                    for batch_image_features, batch_image_sizes in zip(image_features, image_sizes):
                        batch_image_features_list = []
                        for cur_image_feature, cur_image_size in zip(batch_image_features, batch_image_sizes):
                            base_image_feature = cur_image_feature[:width*height, :]
                            image_feature = cur_image_feature[width*height:, :]
                            if image_feature.shape[0] != 0:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    cur_image_size,
                                    self.config.grid_pinpoints,
                                    self.config.vit_crop_size
                                )
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) # (num_patch_H, num_patch_W, H, W, C)
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # (C, num_patch_H, H, num_patch_W, W)
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3) # (C, num_token_H, num_token_W)
                                image_feature = unpad_image(image_feature, cur_image_size) # (C, num_token_H_unpad, num_token_W_unpad)
                                input_shape = (image_feature.shape[-2], image_feature.shape[-1])
                                subimage_tokens = np.prod(input_shape)
                                
                                # adaptive avg 2d pool for reducing token num
                                max_subimage_tokens = self.config.max_vision_token-width*height
                                if subimage_tokens > max_subimage_tokens:
                                    aspect_ratio = input_shape[0] / input_shape[1]
                                    output_shape = (
                                        int((max_subimage_tokens/aspect_ratio)**0.5*aspect_ratio),
                                        int((max_subimage_tokens/aspect_ratio)**0.5)
                                    )
                                    m = nn.AdaptiveAvgPool2d(output_shape)
                                    image_feature = m(image_feature)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            else:
                                image_feature = cur_image_feature
                            batch_image_features_list.append(image_feature)
                        new_image_features.append(batch_image_features_list)

                    image_features = new_image_features

        else:
            image_features = self.encode_images(images).to(self.device)


        # Step2: Iterate through each sample in the batch, insert image embedings into input_embeds
        #        and filling labels, attention mask at the same time. Finally, get `new_input_embed`,
        #        `new_labels`, new_attention_mask`.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask.bool())]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask.bool())]
        attention_mask = [cur_attention_mask[cur_attention_mask.bool()] for cur_attention_mask in attention_mask]
        new_input_embeds = []
        new_labels = []
        new_attention_mask = []
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_batch_image_idx = 0
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            # Step2-1: If this piece of data is pure text, then concat a dummy image to ensure the whole compute graph is same on all device
            if num_images == 0: 
                if getattr(self.config, "eagle_vision_tower", None) is not None:
                    if getattr(self.config, 'only_navit', False):
                        cur_image_features = qwen2vl_image_features[batch_idx][cur_batch_image_idx]
                    else:
                        siglip_feat = image_features[batch_idx][cur_batch_image_idx]
                        try:
                            qwen2vl_feat = qwen2vl_image_features[batch_idx][cur_batch_image_idx]
                            cur_image_features = torch.cat((siglip_feat, qwen2vl_feat), dim=0)
                        except Exception as e:
                            print(e)
                            print("only siglip feature:", siglip_feat.shape)
                            cur_image_features = siglip_feat
                else:
                    cur_image_features = image_features[batch_idx][cur_batch_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features.squeeze(0)[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_attention_mask.append(attention_mask[batch_idx])
                cur_batch_image_idx += 1
                continue
            
            # Step2-2: Split input_ids, labels, attention_mask by IMAGE_TOKEN_INDEX
            cur_input_ids_noim, cur_labels_noim, cur_attention_mask_noim = [], [], []
            cur_labels = labels[batch_idx]
            cur_attention_mask = attention_mask[batch_idx]
            cur_img_attention_mask = [
                attention_mask[batch_idx][i].item()
                for i in torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            ]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] 
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_attention_mask_noim.append(cur_attention_mask[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = list(torch.split(cur_input_embeds, split_sizes, dim=0))# get text features

            # Step2-3: Insert image embedings
            cur_new_input_embeds, cur_new_labels, cur_new_attention_mask = [], [], []
            for i in range(num_images + 1): # to add multimodal feature internal the text feature
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_attention_mask.append(cur_attention_mask_noim[i])
                if i < num_images:
                    if getattr(self.config, "eagle_vision_tower", None) is not None:
                        if getattr(self.config, 'only_navit', False):
                            cur_image_features = qwen2vl_image_features[batch_idx][cur_batch_image_idx]
                        else:
                            siglip_feat = image_features[batch_idx][cur_batch_image_idx]
                            try:
                                qwen2vl_feat = qwen2vl_image_features[batch_idx][cur_batch_image_idx]
                                cur_image_features = torch.cat((siglip_feat, qwen2vl_feat), dim=0)
                            except Exception as e:
                                print(e)
                                print("only siglip feature:", siglip_feat.shape)
                                cur_image_features = siglip_feat
                    else:
                        cur_image_features = image_features[batch_idx][cur_batch_image_idx]
                    cur_batch_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_attention_mask.append(torch.full((cur_image_features.shape[0],), True, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype))

            # Step2-4: Concat image embedings and text embedings
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_attention_mask = torch.cat(cur_new_attention_mask)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_attention_mask.append(cur_new_attention_mask)

        # Step3: Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_attention_mask = [x[:tokenizer_model_max_length] for x in new_attention_mask]

        # Step4: Pad and stack input_embeds, labels, attention_mask
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask_padded = torch.zeros((batch_size, max_len), dtype=new_attention_mask[0].dtype, device=new_attention_mask[0].device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_attention_mask) in enumerate(zip(new_input_embeds, new_labels, new_attention_mask)):
            cur_len = cur_new_embed.shape[0]
            if self.get_padding_method() == 'left':
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask_padded[i, -cur_len:] = cur_attention_mask
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask_padded[i, :cur_len] = cur_attention_mask
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_labels = new_labels_padded if _labels is not None else None
        new_attention_mask = new_attention_mask_padded if _attention_mask is not None else None
        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, new_attention_mask, past_key_values, new_input_embeds, new_labels


class ValleyQwen2Model(ValleyMetaModel, Qwen2Model):
    config_class = ValleyConfig
    def __init__(self, config: Qwen2Config):
        super(ValleyQwen2Model, self).__init__(config)


class ValleyQwen2ForCausalLM(Qwen2ForCausalLM, ValleyMetaForCausalLM):
    config_class = ValleyConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ValleyQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def _update_model_kwargs_for_generation(
        self,
        outputs: CausalLMOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        new_model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, 
            model_kwargs, 
            is_encoder_decoder, 
            num_new_tokens
        )
        """
        Set model_kwargs["attention_mask"] to the expanded `attention_mask` in
        the `prepare_inputs_labels_for_multimodal` function to ensure the 
        correctness of the generate behavior when `use_cache` is enabled.
        """
        if not is_encoder_decoder:
            if "attention_mask" in new_model_kwargs:
                attention_mask = outputs.attention_mask
                new_model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        return new_model_kwargs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        image_sizes: Optional[List[List[int]]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='mean')
            bs = shift_labels.shape[0]
            shift_labels = shift_labels.to(shift_logits.device)
            loss = torch.stack([loss_fct(shift_logits[i], shift_labels[i]) for i in range(bs)])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        res =  CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        res.attention_mask = attention_mask
        return res

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "image_sizes": kwargs.get("image_sizes", None),
                "pixel_values": kwargs.get("pixel_values", None),
                "pixel_values_videos": kwargs.get("pixel_values_videos", None),
                "image_grid_thw": kwargs.get("image_grid_thw", None),
                "video_grid_thw": kwargs.get("video_grid_thw", None),
            }
        )
        return model_inputs


class CustomValleyQwen2ForCausalLM(ValleyQwen2ForCausalLM):
    def __init__(self, config, threshold=-1.5):
        super().__init__(config)
        self.threshold = threshold

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        image_sizes: Optional[List[List[int]]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndMask]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
            return_dict=True,  # Always return dict for compatibility
            image_sizes=image_sizes,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,                
        )
        hidden_states = outputs.hidden_states[-1]
        modified_hidden_states = self.modify_hidden_states(hidden_states)
        logits = self.lm_head(modified_hidden_states)

        return CausalLMOutputWithPastAndMask(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    def modify_hidden_states(self, hidden_states):
        if self.threshold is not None:
            threshold = self.threshold
            hidden_states = torch.relu(hidden_states - threshold) + threshold
        return hidden_states

    

AutoConfig.register("valley", ValleyConfig)
AutoModelForCausalLM.register(ValleyConfig, CustomValleyQwen2ForCausalLM)