"""
Template for customized models through ForCausalLM class

Modified to extract the attentions and hidden states
from the final transformer layer of the model.

For correlation analysis between mean of the head-max attention
and the number of hidden states below a threshold.
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import re
import sys
import numpy as np
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath('/data/students/earl/llava-dissector/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

def extract_bounding_boxes(response):
    pattern = r'(?:[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)])|(?:<box>\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*</box>)'
    matches = re.findall(pattern, response)
    bounding_boxes = []
    for match in matches:
        coords = [c for c in match if c != '']
        if len(coords) == 4:
            bounding_boxes.append([float(c) for c in coords])
    return bounding_boxes

def eval_deepseek(args):
    device = args.device if hasattr(args, "device") else "cuda:0"
    model_path = args.model_path

    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    ).eval()

    if args.threshold == 'None' or args.threshold is None:
        vl_gpt.threshold = None
        threshold_c = -1000
    else:
        vl_gpt.threshold = float(args.threshold)
        threshold_c = float(args.threshold)

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
            question_text = q["text"]
            question_id = q.get("question_id", None)

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{question_text}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(vl_gpt.device)

            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            with torch.inference_mode():
                outputs = vl_gpt.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
                generated_ids = outputs.sequences

                # Forward pass for hidden states and attentions
                with torch.inference_mode():
                    forward_outputs = vl_gpt.language(
                        input_ids=generated_ids,
                        attention_mask=(generated_ids != tokenizer.pad_token_id),
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True
                    )
                    hidden_state_data = forward_outputs.hidden_states[-1][0].to(torch.float32).cpu().detach().numpy()
                    attention_data = forward_outputs.attentions[-1][0].to(torch.float32).cpu().detach().numpy()
                    # Debug print
                    #print(f'data type of attentions: {type(forward_outputs.attentions)}')
                    #print(f'forward_outputs.attentions.shape: {forward_outputs.attentions[0].shape}')
                    print(f'attention_data shape: {attention_data.shape}, hidden_state_data shape: {hidden_state_data.shape}')

                    # Compute statistics
                    below_threshold_count = (hidden_state_data < threshold_c).sum().item()
                    below_threshold_counts.append(below_threshold_count)  # Store the count

                    # Compute the maximum attention value for each head (across all tokens), then take the mean across heads
                    # attention_data shape: (num_heads, seq_len, seq_len)
                    # Step 1: For each head and each column, get the max over rows (axis=1)
                    # This gives shape (num_heads, seq_len)
                    max_per_column = attention_data.max(axis=1)

                    # Step 2: Mean across heads (axis=0), result shape (seq_len,)
                    mean_max_per_column = max_per_column.mean(axis=0)

                    # If you want a single scalar (mean over all columns as well):
                    mean_attention_weight = float(mean_max_per_column.mean())
                    mean_attention_weights_list.append(mean_attention_weight)

                    # Debug print
                    print(f'Question ID: {question_id}, Below Threshold Count: {below_threshold_count}, Mean Attention Weight: {mean_attention_weight}')
                    # Free memory
                    del hidden_state_data, attention_data#, max_attention_weights
                    torch.cuda.empty_cache()

                    correlation = {
                        "question_id": question_id,
                        "below_threshold_count": below_threshold_count,
                        "mean_attention_weights": mean_attention_weight
                    }
                    all_correlations.append(correlation)

                # Generate the final output
                response = tokenizer.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)
                print(f"Response: {response}")

            bounding_boxes = extract_bounding_boxes(response)
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": question_id,
                "prompt": question_text,
                "text": response,
                "bounding_boxes": bounding_boxes,
                "answer_id": ans_id,
                "model_id": os.path.basename(args.model_path),
                "metadata": f'model.threshold: {vl_gpt.threshold}'
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

    with open(correlations_file, "w") as corr_file:
        json.dump(all_correlations, corr_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=str, default=None)
    args = parser.parse_args()
    eval_deepseek(args)