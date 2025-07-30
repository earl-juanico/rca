from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizer
import sys
sys.path.append('/data/students/earl/llava-dissector/POINTS-1-5-Qwen-2-5-7B-Chat/WePOINTS/wepoints')
from wepoints.models.vision.qwen2_vl_navit import Qwen2VisionTransformerForNavitPOINTS  # noqa

try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchMerger
except ImportError:
    print('Please upgrade transformers to version 4.46.3 or higher')

from .configuration_pointsv15_chat import POINTSV15ChatConfig
from .modeling_llama import CustomLlamaForCausalLM

try:
    from wepoints.models import Qwen2VisionTransformerForNavitPOINTS
except ImportError:
    print('Please install WePOINTS, and refer to https://github.com/WePOINTS/WePOINTS')


class POINTSV15ChatModel(PreTrainedModel, GenerationMixin):
    config_class = POINTSV15ChatConfig
    _no_split_modules = ["CustomLlamaLayer",
                         "Qwen2VisionTransformerPretrainedModel"]

    """Chat model for POINTSv1.5."""

    def __init__(self, config: POINTSV15ChatConfig) -> None:
        super().__init__(config)
        self.llm = CustomLlamaForCausalLM(config.llm_config)
        self.vision_encoder = Qwen2VisionTransformerForNavitPOINTS._from_config(
            config.vision_config, attn_implementation="flash_attention_2"
        )
        self.vision_projector = PatchMerger(config.llm_config.hidden_size,
                                            context_dim=1280)

    def process_images(self, images: torch.Tensor, 
                       image_grid_thws: List[list]) -> torch.Tensor:
        """Obtain image features from the vision encoder."""
        image_features = self.vision_encoder(images, grid_thw=image_grid_thws)
        image_features = self.vision_projector(image_features)
        return image_features

    def construct_prompt(self, messages: List[dict],
                         image_processor: Qwen2VLImageProcessor) -> Tuple[str, List[torch.Tensor], List[list]]:
        """Construct the prompt for the chat model."""
        images = []
        image_grid_thws = []
        reconstructed_messages = []
        for message in messages:
            role = message['role']
            content_from_role = ''
            for item in message['content']:
                if item['type'] == 'text':
                    content_from_role += item['text']
                elif item['type'] == 'image':
                    image_path = item['image']
                    image = Image.open(image_path).convert('RGB')
                    image_data = image_processor(images=image)
                    pixel_values = image_data['pixel_values']
                    image_grid_thw = image_data['image_grid_thw']
                    # pixel_values is a tensor or list of tensors
                    if isinstance(pixel_values, torch.Tensor):
                        images.append(pixel_values)
                    else:
                        images.extend(pixel_values)
                    image_grid_thws.append(image_grid_thw)
                    seq_len = int(image_grid_thw[0][1] * image_grid_thw[0][2] / 4)
                    content_from_role += '<|vision_start|>' + '<|image_pad|>' * seq_len + '<|vision_end|>' + '\n'
            reconstructed_messages.append({
                'role': role,
                'content': content_from_role
            })
        prompt = self.apply_chat_template(reconstructed_messages)
        return prompt, images, image_grid_thws

    def apply_chat_template(self, messages: List[dict]) -> str:
        """Apply the chat template to the input messages."""
        role_prefix_mapping = {
            'user': '<|im_start|>user\n',
            'assistant': '<|im_start|>assistant\n'
        }
        role = 'user'
        prompt = ''
        for message in messages:
            role = message['role']
            content = message['content']
            prompt += role_prefix_mapping[role] + content + '<|im_end|>\n'
        if role == 'user':
            prompt += '<|im_start|>assistant\n'
        return prompt

    @torch.no_grad()
    def chat(self, 
             messages: List[dict],
             tokenizer: PreTrainedTokenizer,
             image_processor: object,
             generation_config: dict = None) -> str:
        """Generate a response to the input prompt."""
        prompt, images, image_grid_thws = self.construct_prompt(
            messages, image_processor
        )
        # Stack images into a batch tensor
        if len(images) > 0 and isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)
        else:
            images = torch.tensor(images)
        images = images.to(self.vision_encoder.device).to(self.vision_encoder.dtype)
        # Concatenate image_grid_thws if needed
        if isinstance(image_grid_thws[0], (list, np.ndarray)):
            image_grid_thws = np.concatenate(image_grid_thws, axis=0)
        image_grid_thws = torch.from_numpy(np.array(image_grid_thws)).to(self.vision_encoder.device).long()
        image_features = self.vision_encoder(images, grid_thw=image_grid_thws)
        image_features = self.vision_projector(image_features)
        model_inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if generation_config is None:
            generation_config = {}
        generation_config.update(
            {
                'eos_token_id': eos_token_id,
            }
        )
        outputs = self.generate(
            input_ids=input_ids,
            image_grid_thws=image_grid_thws,
            attention_mask=attention_mask,
            image_features=[image_features],
            image_token_id=image_token_id,
            **generation_config
        )
        # If outputs is a tuple or dict, get the first element or 'sequences'
        if isinstance(outputs, dict) and 'sequences' in outputs:
            outputs = outputs['sequences']
        elif isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def _split_input_ids(self, input_ids, special_token):
        special_pos = input_ids == special_token
        pos = (special_pos[:-1] != special_pos[1:]).nonzero() + 1
        if pos.shape[0] % 2 != 0:
            pos = torch.cat([torch.tensor([[0]]).to(pos.device), pos])
        pos = pos.reshape(-1, 2).tolist()
        return pos

    def generate(self,
                 input_ids: torch.LongTensor,
                 image_grid_thws: torch.LongTensor,
                 attention_mask: torch.LongTensor,
                 image_features: List[torch.Tensor],
                 image_token_id: int,
                 generation_config: Optional[dict] = None,
                 output_hidden_states: Optional[bool] = None,
                 return_dict: Optional[bool] = None,
                 **generate_kwargs) -> torch.LongTensor:
        input_embeddings = self.llm.lm.embed_in(input_ids)
        batch_size = input_ids.shape[0]
        assert len(image_features) == batch_size or len(image_features) == 1
        for i in range(batch_size):
            pos = self._split_input_ids(input_ids[i], image_token_id)
            assert len(pos) == len(image_grid_thws)
            image_pos = [
                int(image_grid_thw[1] * image_grid_thw[2] / 4)
                for image_grid_thw in image_grid_thws
            ]
            image_pos.insert(0, 0)
            image_pos = np.cumsum(image_pos)
            for j, (start, end) in enumerate(pos):
                # If image_features is a list of batch, index accordingly
                if len(image_features) == batch_size:
                    feats = image_features[i]
                else:
                    feats = image_features[0]
                input_embeddings[i, start:end] = feats[image_pos[j]:image_pos[j+1]]
        outputs = self.llm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs
        )
        return outputs

class RCACustomLlamaForCausalLM(CustomLlamaForCausalLM):
    def __init__(self, config, modify_hidden_states=None):
        super().__init__(config)
        self.modify_hidden_states = modify_hidden_states

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, output_hidden_states=True, **kwargs)
        if self.modify_hidden_states is not None and hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
            modified_hidden_states = self.modify_hidden_states(hidden_states)
            logits = self.embed_out(modified_hidden_states) # lm_head is embed_out
            outputs.logits = logits
        return outputs 
    
class CustomPOINTSV15ChatModel(POINTSV15ChatModel):
    def __init__(self, config: POINTSV15ChatConfig, threshold=-1.5) -> None:
        super().__init__(config)
        self.threshold = threshold
        self.llm = RCACustomLlamaForCausalLM(config.llm_config, modify_hidden_states=self.modify_hidden_states)
        self.llm.modify_hidden_states = self.modify_hidden_states

    def modify_hidden_states(self, hidden_states):
        if self.threshold is not None:
            threshold = self.threshold
            hidden_states = torch.relu(hidden_states - threshold) + threshold
        return hidden_states
