###########################################################################################################
#####                                biuzz  2023   待修改                                             #####
###########################################################################################################
import argparse
import itertools
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask

from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from typing import Optional, Tuple, Union
from datasets import OpenImagesDataset
from timm.models.vision_transformer import Attention, Mlp


# Text Adapter
class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(7):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim),
                                         nn.LayerNorm(output_dim),
                                         nn.LeakyReLU(),
                                         nn.Linear(output_dim, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim),
                                                        nn.LayerNorm(output_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(output_dim, output_dim)))
    def forward(self, embs):

        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states = hidden_states + (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states
    

class MapperLocal(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(MapperLocal, self).__init__()

        for i in range(7):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim),
                                         nn.LayerNorm(output_dim),
                                         nn.LeakyReLU(),
                                         nn.Linear(output_dim, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim),
                                                        nn.LayerNorm(output_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(output_dim, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states = hidden_states + (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1]).to('cuda')


    # 替换 inj_embedding
    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
            new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]       # 替换[bsz, 6~77], shape为(71, 768), 即替换了71个值
            new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]         # 替换[bsz, 5~6], 其中5是伪词token的位置, shape为(1, 768), 即替换了1个值

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
    context = encoder_hidden_states
    hidden_states_local = hidden_states.clone() 
    hidden_states_update = hidden_states.clone() 

    if context is not None:
        context_tensor = context["CONTEXT_TENSOR"]
    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape
    
    query = self.to_q(hidden_states)

    if context is not None:
        key = self.to_k_global(context_tensor)
        value = self.to_v_global(context_tensor)
    else:
        key = self.to_k(context_tensor)
        value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)
    
    if context is not None and "LOCAL" in context:
        # Perform cross attention with the local context
        query_local = self.to_q(hidden_states_local)
        key_local = self.to_k_local(context["LOCAL"])
        value_local = self.to_v_local(context["LOCAL"])

        query_local = self.reshape_heads_to_batch_dim(query_local)
        key_local = self.reshape_heads_to_batch_dim(key_local)
        value_local = self.reshape_heads_to_batch_dim(value_local)

        attention_scores_local = torch.matmul(query_local, key_local.transpose(-1, -2))
        attention_scores_local = attention_scores_local * self.scale
        attention_probs_local = attention_scores_local.softmax(dim=-1)

        # To extract the attmap of learned [w]
        index_local = context["LOCAL_INDEX"]
        index_local = index_local.reshape(index_local.shape[0], 1).repeat((1, self.heads)).reshape(-1)
        attention_probs_clone = attention_probs.clone().permute((0, 2, 1))
        attention_probs_mask = attention_probs_clone[torch.arange(index_local.shape[0]), index_local]
        # Normalize the attention map
        attention_probs_mask = attention_probs_mask.unsqueeze(2) / attention_probs_mask.max()

        if "LAMBDA" in context:
            _lambda = context["LAMBDA"]
        else:
            _lambda = 1

        attention_probs_local = attention_probs_local * attention_probs_mask * _lambda
        hidden_states += torch.matmul(attention_probs_local, value_local)
    
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states



logger = logging.getLogger(__name__)


def save_progress(mapper, args, save_name, step=None):
    logger.info("Saving embeddings")

    state_dict = mapper.state_dict()

    if step is not None:
        torch.save(state_dict, os.path.join(args.output_dir, f"{str(save_name)}_{str(step).zfill(6)}.pt"))
    else:
        torch.save(state_dict, os.path.join(args.output_dir, f"{str(save_name)}.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="",
        required=True,
        help="Path to pretrained model or model identifier from vae.",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        default="",
        required=True,
        help="Path to pretrained model or model identifier from unet.",
    )
    parser.add_argument(
        "--ddim_path",
        type=str,
        default="",
        required=True,
        help="Path to pretrained model or model identifier from ddim.",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default="",
        required=True,
        help="Path to pretrained model or model identifier from clip.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--valid_data_dir", type=str, default=None, required=True, help="A folder containing the valid data."
    )
    parser.add_argument(
        "--global_mapper_path", type=str, default=None, help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--local_mapper_path", type=str, default=None,
        help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


@torch.no_grad()
def validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae, device, guidance_scale, token_index=0, seed=42, llambda=1):
    scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64), generator=generator,
        )

    latents = latents.to(example["pixel_values_clip"])
    scheduler.set_timesteps(20)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]
    image = F.interpolate(example["pixel_values_clip"].to("cuda"), (224, 224), mode='bilinear')

    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16], image_features[2][18], image_features[2][20]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    image_embeddings = torch.stack(image_embeddings)

    inj_embedding_local = mapper_local(image_embeddings)

    inj_embedding = mapper(image_embeddings)

    if token_index != 'full':
        token_index = int(token_index)
        inj_embedding_first_word = inj_embedding[:, token_index:token_index + 1, :]

    with torch.no_grad():
        encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                              "inj_embedding": inj_embedding_first_word,
                                              "inj_index": placeholder_idx})[0]                                


    for t in tqdm(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t).to("cuda")
        
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
                "LOCAL": inj_embedding_local,
                "LOCAL_INDEX": placeholder_idx.detach(),
                "LAMBDA": llambda
            }
        ).sample

        latent_model_input = scheduler.scale_model_input(latents, t).to("cuda")

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
            }
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred.to("cuda"), t.to("cuda"), latents.to("cuda")).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # If passed along, set the training seed now.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.clip_path:
        version = args.clip_path
        tokenizer = CLIPTokenizer.from_pretrained(version)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(version)

    # replace the forward method of the text encoder to inject the word embedding
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text


    image_encoder = CLIPVisionModel.from_pretrained(version) 

    # Load image adapter and text adapter
    mapper = Mapper(input_dim=1024, output_dim=768)
    mapper_local = MapperLocal(input_dim=1024, output_dim=768)

    version2 = args.vae_path
    vae = AutoencoderKL.from_pretrained(version2)

    version3 = args.unet_path
    unet = UNet2DConditionModel.from_pretrained(version3)

    # replace the forward method of the crossattention to finetune the to_k and to_v layers
    # 避免在所有cross attn中注入强化特征, 这样虽然会提高id保真度但会极大地削弱可编辑性, 所以仅需在attn2中注入增强特征即可
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention           

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            to_k_global.weight.data = _module.to_k.weight.data.clone().to("cuda")
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            to_v_global.weight.data = _module.to_v.weight.data.clone().to("cuda")
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            to_k_local = nn.Linear(shape[1], shape[0], bias=False)
            to_k_local.weight.data = _module.to_k.weight.data.clone().to("cuda")
            mapper_local.add_module(f'{_name.replace(".", "_")}_to_k', to_k_local)
            _module.add_module('to_k_local', to_k_local)

            to_v_local = nn.Linear(shape[1], shape[0], bias=False)
            to_v_local.weight.data = _module.to_v.weight.data.clone().to("cuda")
            mapper_local.add_module(f'{_name.replace(".", "_")}_to_v', to_v_local)
            _module.add_module('to_v_local', to_v_local)

            if args.global_mapper_path is None:
                _module.add_module('to_k_global', to_k_global)
                _module.add_module('to_v_global', to_v_global)

            if args.local_mapper_path is None:
                _module.add_module('to_k_local', to_k_local)
                _module.add_module('to_v_local', to_v_local)
    
    if args.global_mapper_path is not None:
        mapper.load_state_dict(torch.load(args.global_mapper_path, map_location='cpu'))
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name: continue
                _module.add_module('to_k_global', getattr(mapper, f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_global', getattr(mapper, f'{_name.replace(".", "_")}_to_v'))          

    if args.local_mapper_path is not None:
        mapper_local.load_state_dict(torch.load(args.local_mapper_path, map_location='cpu'))
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name: continue
                _module.add_module('to_k_local', getattr(mapper_local, f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_local', getattr(mapper_local, f'{_name.replace(".", "_")}_to_v'))


    # Freeze vae and unet, encoder
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())

    # Unfreeze the mapper
    unfreeze_params(mapper.parameters())
    unfreeze_params(mapper_local.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": itertools.chain(mapper.parameters(), mapper_local.parameters()), "lr": args.learning_rate},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.ddim_path)

    train_dataset = OpenImagesDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        set="train",
    )

    valid_dataset = OpenImagesDataset(
        data_root=args.valid_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        set="valid",
        mask_name="valid_mask",
        jsonl_name="valid_captions.jsonl",
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False)
    
    for batch_valid in valid_dataloader:
        batch_valid = batch_valid

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Move vae, unet, and encoders to device
    vae.to("cuda")
    unet.to("cuda")
    image_encoder.to("cuda")
    text_encoder.to("cuda")
    mapper.to("cuda")
    mapper_local.to("cuda")
    
    # Keep vae, unet and image_encoder in eval model as we don't train these
    vae.eval()
    unet.eval()
    image_encoder.eval()
    text_encoder.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    progress_bar.set_description("Steps")
    global_step = 0
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_train_epochs):
        mapper.train()
        mapper_local.train()

        for batch in train_dataloader:

            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to("cuda")).latent_dist.sample().detach()        # (bs, 4, 64, 64)
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn(latents.shape).to(latents.device)       # (bs, 4, 64, 64)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # mask
            # mask = batch["mask"].permute(0, 3, 1, 2)
            # mask = F.interpolate(mask.to("cuda").float(), (16, 16), mode='nearest')
            # mask = mask[:, 0].reshape(mask.shape[0], -1, 1)

            noisy_latents_res = []
            for i in range(noisy_latents.shape[0]):
                noisy_latents_res.append(noisy_latents[i])

            noisy_latents_res = torch.stack(noisy_latents_res, dim=0).to(device="cuda")

            placeholder_idx = batch["index"].to("cuda")
            image = F.interpolate(batch["pixel_values_clip"].to("cuda"), (224, 224), mode='bilinear')

            with torch.no_grad():
                image_features = image_encoder(image, output_hidden_states=True)


            # [0]是主要初始概念特征, [4]~[16]是无关特征
            image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16], image_features[2][18], image_features[2][20]]
            image_embeddings = [emb.detach() for emb in image_embeddings]
            image_embeddings = torch.stack(image_embeddings)

            # use mix precision
            with torch.cuda.amp.autocast():

                # Local mapper
                inj_embedding_local = mapper_local(image_embeddings)    
                # inj_embedding_local = inj_embedding_local * mask

                # Global mapper
                inj_embedding = mapper(image_embeddings)[:, 0:1, :]

                # Get the text embedding for conditioning
                # text_encoder是一个自适应的编码器, 无论inj_embedding的第二个维度是多少, 其输出的张量维度都是 (bs, 77, 768)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder({'input_ids': batch["input_ids"].to("cuda"),
                                                          "inj_embedding": inj_embedding,
                                                          "inj_index": placeholder_idx.detach()})[0]
                
                # noise_pred: (4, 4, 64, 64)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states={
                    "CONTEXT_TENSOR": encoder_hidden_states,
                    "LOCAL": inj_embedding_local,
                    "LOCAL_INDEX": placeholder_idx.detach()
                }).sample

                loss_mle = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                # regularization loss
                loss_reg_text = torch.mean(torch.abs(inj_embedding)) * 0.01
                loss_reg_image = torch.mean(torch.abs(inj_embedding_local)) * 0.001

                total_loss = loss_mle + loss_reg_text + loss_reg_image 


            # total_loss.backward()
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            lr_scheduler.step()
            optimizer.zero_grad()
            scaler.update()


            global_step += 1
            if global_step % args.save_steps == 0:
                save_progress(mapper, args, "mapper", global_step)
                save_progress(mapper_local, args, "mapper_local", global_step)
                if global_step < 50000:
                    syn_images = validation(batch, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae, batch["pixel_values_clip"].device, 5)
                    gt_images = [th2image(img) for img in batch["pixel_values"]]
                else:
                    syn_images = validation(batch_valid, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae, batch["pixel_values_clip"].device, 5)
                    gt_images = [th2image(img) for img in batch_valid["pixel_values"]]
                
                img_list = []
                for syn, gt in zip(syn_images, gt_images):
                    img_list.append(np.concatenate((np.array(syn), np.array(gt)), axis=1))
                img_list = np.concatenate(img_list, axis=0)
                Image.fromarray(img_list).save(os.path.join(args.output_dir, f"{str(global_step).zfill(5)}.jpg"))

            logs = {"loss_mle": loss_mle.detach().item(), "loss_reg_text": loss_reg_text.detach().item(), "loss_reg_image": loss_reg_image.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            

            if global_step >= args.max_train_steps:
                break





if __name__ == "__main__":
    main()
