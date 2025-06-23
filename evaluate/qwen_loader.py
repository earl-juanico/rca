# conda activate evalqwen
# Applicable for Qwen2.5-VL-7B and WeThink-Qwen2.5-VL-7B

from tqdm import tqdm
from PIL import Image
import torch
import json
import os
import shortuuid
import gc

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from qwen_vl_utils import process_vision_info

from typing import Optional, List, Tuple, Union

class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
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
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
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
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
        hidden_states = outputs.hidden_states[-1]
        modified_hidden_states = self.modify_hidden_states(hidden_states)
        logits = self.lm_head(modified_hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if outputs.attentions is not None else [],
            rope_deltas=outputs.rope_deltas,
        )

    def modify_hidden_states(self, hidden_states):
        if self.threshold is not None:
            threshold = self.threshold
            hidden_states = torch.relu(hidden_states - threshold) + threshold
        return hidden_states

def get_model_processor(model_dir, device='cuda:0'):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    ).eval() #to(device).eval()
    return model, processor

def extract_bounding_boxes(response):
    # Example regex for extracting bounding boxes
    import re
    pattern = r'(?:[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*[\]\)]|</box>))|(?:<box>\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*</box>)'
    matches = re.findall(pattern, response)
    bboxes = []
    for match in matches:
        nums = [x for x in match if x != '']
        if len(nums) == 4:
            bboxes.append([float(x) for x in nums])
    return bboxes

def eval_qwen(args):
    device = args.device if hasattr(args, "device") else "cuda:0"
    model, processor = get_model_processor(args.model_path, device)
    if args.threshold == 'None' or args.threshold is None:
        threshold_c = -1.5
    else:
        threshold_c = float(args.threshold)

    # Load questions
    with open(args.question_file, "r") as f:
        questions = [json.loads(line) for line in f]

    answers_file = os.path.expanduser(args.answers_file)
    correlations_file = os.path.join(os.path.dirname(answers_file), "correlations.json")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    all_correlations = []

    for q in tqdm(questions, desc="Evaluating"):
        image_path = os.path.join(args.image_folder, q["image"])
        if image_path.startswith("http"):
            import io, requests
            image = Image.open(io.BytesIO(requests.get(image_path).content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        question_text = q["text"]
        question_id = q.get("question_id", None)

        messages = [
            {
                'role': 'system',
                'content': (
                    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n"
                    "The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=False,#True,
                output_attentions=False,#True
            )
            '''
            # Hidden states and attentions extraction
            hidden_states = outputs.hidden_states[-1][-1].detach()
            attentions = outputs.attentions[-1][-1].detach()
            if isinstance(attentions, tuple):
                attentions = attentions[0].detach()
            # Detach before any further processing to avoid memory blow-up
            below_threshold_count = (hidden_states < threshold_c).sum().item()
            max_attention_weights = attentions.detach().to(torch.float32).max(dim=1).values.max(dim=2).values.cpu().numpy()
            mean_attention_weight = float(max_attention_weights.mean())
            correlation = {
                "question_id": question_id,
                "below_threshold_count": below_threshold_count,
                "mean_attention_weights": mean_attention_weight
            }
            all_correlations.append(correlation)
            # Free memory
            del hidden_states, attentions, max_attention_weights
            torch.cuda.empty_cache()
            '''
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            


        bounding_boxes = extract_bounding_boxes(output_text)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": question_id,
            "prompt": question_text,
            "text": output_text,
            "bounding_boxes": bounding_boxes,
            "answer_id": ans_id,
            "model_id": os.path.basename(args.model_path),
            "metadata": {}
        }) + "\n")

    ans_file.close()

    # Save all correlations to a file
    #with open(correlations_file, "w") as corr_file:
    #    json.dump(all_correlations, corr_file, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=str, default=None)
    args = parser.parse_args()
    eval_qwen(args)