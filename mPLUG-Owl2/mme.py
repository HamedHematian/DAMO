from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

import torch
from PIL import Image
from transformers import TextStreamer
import os
import json

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import logging
 
logging.disable(logging.WARNING)

model_path = 'mplug-owl2-llama2-7b'
model_name = get_model_name_from_path(model_path)
# print(model_name)  # mplug-owl2-llama2-7b
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
# print(model)  MPLUGOwl2LlamaForCausalLM


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--output_hidden_states", type=bool, default=False)
parser.add_argument("--noise_step", type=int, default=500)
parser.add_argument("--cd_alpha", type=float, default=1)
parser.add_argument("--cd_beta", type=float, default=0.1)
parser.add_argument("--use_cd", action='store_true', default=False)
parser.add_argument("--use_dola", action='store_true', default=False)
parser.add_argument("--use_opera", action='store_true', default=False)
args = parser.parse_args()


print(args)



def model_eval(query, image_file):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles
    ##Process Image
    image = Image.open(image_file).convert('RGB')
    # max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    # We modify the parameters here, we set the max_edge to 224.
    max_edge = 224
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    # print(image_tensor.shape)  # 1, 3, 448, 448
    ##Process query
    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    if args.use_cd:
        from vcd_utils.vcd_add_noise import add_diffusion_noise
        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                images_cd=(image_tensor_cd.unsqueeze(0).cuda().to(dtype=image_tensor.dtype) if image_tensor_cd is not None else None),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_hidden_states=True,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
            )

    elif args.use_dola:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_hidden_states=True,
                dola_decoding=True,
                mature_layer=32,
                base_layer=None,
                candidate_premature_layers=[0,2,4,6,8,10,12,14],
                relative_top= 0,
                contrastive_decoding=None,
                student_model = None,
            )

    elif args.use_opera:
        ### for OPERA
        key_position = {
            "image_start": 5,
            "image_end": 69,
            "response_start": input_ids.size(1) + 65-1,
            }
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=args.output_hidden_states,
                num_beams=4,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=50,
                threshold=25,
                num_attn_candidates=1,
                penalty_weights=1,
                key_position=key_position,
            )

        
    else:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_hidden_states=True,
            )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return outputs



output_dir = ""
image_folder_prefix = "MME_Benchmark_release_version"

folders_and_files = [
    {
        "image_folder": "artwork/images",
        "output_file_path": f"eval_tool/{output_dir}/artwork.txt"
    },
    {
        "image_folder": "celebrity/images",
        "output_file_path": f"eval_tool/{output_dir}/celebrity.txt"
    },
    {
        "image_folder": "code_reasoning",
        "output_file_path": f"eval_tool/{output_dir}/code_reasoning.txt"
    },
    {
        "image_folder": "color",
        "output_file_path": f"eval_tool/{output_dir}/color.txt"
    },
    {
        "image_folder": "commonsense_reasoning",
        "output_file_path": f"eval_tool/{output_dir}/commonsense_reasoning.txt"
    },
    {
        "image_folder": "count",
        "output_file_path": f"eval_tool/{output_dir}/count.txt"
    },
    {
        "image_folder": "existence",
        "output_file_path": f"eval_tool/{output_dir}/existence.txt"
    },
    {
        "image_folder": "landmark/images",
        "output_file_path": f"eval_tool/{output_dir}/landmark.txt"
    },
    {
        "image_folder": "numerical_calculation",
        "output_file_path": f"eval_tool/{output_dir}/numerical_calculation.txt"
    },
    {
        "image_folder": "OCR",
        "output_file_path": f"eval_tool/{output_dir}/OCR.txt"
    },
    {
        "image_folder": "position",
        "output_file_path": f"eval_tool/{output_dir}/position.txt"
    },
    {
        "image_folder": "posters/images",
        "output_file_path": f"eval_tool/{output_dir}/posters.txt"
    },
    {
        "image_folder": "scene/images",
        "output_file_path": f"eval_tool/{output_dir}/scene.txt"
    },
    {
        "image_folder": "text_translation",
        "output_file_path": f"eval_tool/{output_dir}/text_translation.txt"
    },
]

for item in folders_and_files:
    image_folder = os.path.join(image_folder_prefix, item["image_folder"])
    output_file_path = item["output_file_path"]

    with open(output_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        idx = 0
        for line in lines:
            print(f"Processing {idx} in {output_file_path}")
            idx += 1
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                image_file = parts[0]
                question = parts[1]
                image_path = os.path.join(image_folder, image_file)
                output_text = model_eval(question, image_path)
                output_text = output_text.replace('\n', ' ')
                output_text = output_text.replace('\t', ' ')
                output_line = f"{line.strip()}\t{output_text}\n"
                output_file.write(output_line)
    print(f"Finished processing {output_file_path}")

print("All files have been processed!")


# CUDA_VISIBLE_DEVICES=0 python mme.py 