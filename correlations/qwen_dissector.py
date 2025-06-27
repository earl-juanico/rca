"""
Template for customized models through ForConditionalGeneration class

Modified to extract the attentions and hidden states
from the final transformer layer of the model.

For correlation analysis between mean of the head-max attention
and the number of hidden states below a threshold.
"""

from tqdm import tqdm
from PIL import Image
import torch
import json
import os
import shortuuid
import re
import numpy as np

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
            output_attentions=True,#output_attentions,
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
    processor = AutoProcessor.from_pretrained(model_dir)
    model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval() #to(device).eval()
    # Debug
    #processor.tokenizer.padding_side = 'left'
    return model, processor

def extract_bounding_boxes(response):
    # Example regex for extracting bounding boxes
    #pattern = r'(?:[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*[\]\)]|</box>))|(?:<box>\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*</box>)'
    pattern = r'[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]'
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
        threshold_c = -1000
    else:
        threshold_c = float(args.threshold)

    if threshold_c==-1000:
        model.threshold = None
    else:
        model.threshold = threshold_c

    # Load questions
    with open(args.question_file, "r") as f:
        questions = [json.loads(line) for line in f]

    answers_file = os.path.expanduser(args.answers_file)
    correlations_file = os.path.join(os.path.dirname(answers_file), "correlations.json")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    all_correlations = []
    below_threshold_counts = []  # List to store counts for all questions
    mean_attention_weights_list = []  # List to store mean attention weights for all questions


    for q in tqdm(questions, desc="Evaluating"):
        try:
            image_path = os.path.join(args.image_folder, q["image"])
            if image_path.startswith("http"):
                import io, requests
                image = Image.open(io.BytesIO(requests.get(image_path).content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
            question_text = q["text"]
            question_id = q.get("question_id", None)

            messages = [
                # Uncomment the system role for WeThink only
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
                # For correlation analysis purposes
                output_visual = model.generate(
                    **inputs, 
                    max_new_tokens=256, # this limit avoids OOM error #args.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True, 
                    output_hidden_states=True,
                    output_attentions=True
                )

                # Now extract only the last layer
                hidden_state_data = output_visual.hidden_states[-1][-1]#.to(torch.float32).cpu().detach().numpy()
                attention_data = output_visual.attentions[-1][-1]#.to(torch.float32).cpu().detach().numpy()
                # If attentions is a tuple, extract the relevant tensor
                if isinstance(attention_data, tuple):
                    attention_data = attention_data[0]  # The first [0]: extract from tuple; second [0] squeeze batch dim


                # Compute statistics
                below_threshold_count = (hidden_state_data < threshold_c).sum().item()
                below_threshold_counts.append(below_threshold_count)  # Store the count
                max_attention_weights = attention_data.to(torch.float32).max(dim=1).values.max(dim=2).values.detach().cpu().numpy()
                mean_attention_weight = float(max_attention_weights.mean())
                mean_attention_weights_list.append(float(max_attention_weights.mean()))  # Store a single scalar value per sample

                # Free memory
                del hidden_state_data, attention_data, max_attention_weights
                torch.cuda.empty_cache()

                # Debug print
                print(f'\nQuestion ID: {question_id}, Below Threshold Count: {below_threshold_count if below_threshold_count is not None else "None"}, Mean Attention Weight: {mean_attention_weight if mean_attention_weight is not None else "None"}')

                correlation = {
                    "question_id": question_id,
                    "below_threshold_count": int(below_threshold_count),
                    "mean_attention_weights": mean_attention_weight
                }
                all_correlations.append(correlation)

            # Generate the final output
            output_text = processor.batch_decode(
                output_visual.sequences[:, len(inputs["input_ids"]):],  # Slice the output
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
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
                "metadata": f'model.threshold: {model.threshold}'
            }) + "\n")
        except Exception as e:
            print(f"Skipping question_id {q.get('question_id', None)} due to error: {e}")
            continue

        # After the loop, calculate the correlation coefficient
        if below_threshold_counts and mean_attention_weights_list:
            # Debugging: Print the lists
            #print(f"below_threshold_counts: {below_threshold_counts}")
            #print(f"mean_attention_weights_list: {mean_attention_weights_list}")

            # Check if either list has constant values
            if len(set(below_threshold_counts)) == 1 or len(set(mean_attention_weights_list)) == 1:
                print("One of the lists has constant values. Correlation coefficient cannot be calculated.")
            else:
                # Calculate the correlation coefficient
                correlation_coefficient = np.corrcoef(below_threshold_counts, mean_attention_weights_list)[0, 1]
                print(f'length of below_threshold_counts: {len(below_threshold_counts)}')
                print(f'length of mean_attention_weights_list: {len(mean_attention_weights_list)}')
                print(f"Correlation coefficient between below_threshold_count and mean_attention_weights: {correlation_coefficient}")
        else:
            print("Not enough data to calculate correlation coefficient.")

    ans_file.close()

    # Save all correlations to a file
    with open(correlations_file, "w") as corr_file:
        json.dump(all_correlations, corr_file, indent=4)

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