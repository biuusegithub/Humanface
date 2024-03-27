import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json
from tqdm import tqdm


version = '/data/vjuicefs_hz_cv_va/11165350/blip/'
json_path = '/data/vjuicefs_hz_cv_va/11165350/humanface_dataset/captions.jsonl'
img_path = '/data/vjuicefs_hz_cv_va/11165350/humanface_dataset/train' 
mask_path = '/data/vjuicefs_hz_cv_va/11165350/humanface_dataset/mask'


processor = BlipProcessor.from_pretrained(version)
model = BlipForConditionalGeneration.from_pretrained(version, torch_dtype=torch.float16).to("cuda")


image_paths = [os.path.join(img_path, file) for file in os.listdir(img_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]


data = []
for img_path in tqdm(image_paths):
    raw_image = Image.open(img_path).convert('RGB')
    mask_image = os.path.join(mask_path, img_path.split('/')[-1])

    # conditional image captioning
    text = "a photo of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    # print(processor.decode(out[0], skip_special_tokens=True))
    data.append({"raw_path": img_path, 
                 "mask_path": mask_image,
                 "cation": processor.decode(out[0], skip_special_tokens=True),
                 })


with open(json_path, 'w') as jsonl_file:
    for entry in data:
        jsonl_file.write(json.dumps(entry) + '\n')

print('done.')


