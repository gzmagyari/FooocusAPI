import os
import torch
from modules import config, model_management
from apis.utils.img_utils import narray_to_base64img
from modules.patch import PatchSettings, patch_settings
from modules.sample_hijack import sample_hacked
from ldm_patched.modules.sd import load_checkpoint_guess_config
from apis.utils.api_utils import params_to_params
from apis.models.requests import CommonRequest
from apis.models.response import RecordResponse
from apis.models.base import CurrentTask
import datetime
import json
import uuid

class StableDiffusionModel:
    def __init__(self, unet=None, clip=None, vae=None, clip_vision=None, filename=None, vae_filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self.visited_loras = ''

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})

def model_lora_keys_unet(model, key_map={}):
    sdk = model.state_dict().keys()
    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
    return key_map

def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()
    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    LORA_CLIP_MAP = {
        "mlp.fc1": "mlp_fc1",
        "mlp.fc2": "mlp_fc2",
        "self_attn.k_proj": "self_attn_k_proj",
        "self_attn.q_proj": "self_attn_q_proj",
        "self_attn.v_proj": "self_attn_v_proj",
        "self_attn.out_proj": "self_attn_out_proj",
    }
    for b in range(32):
        for c in LORA_CLIP_MAP:
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
    return key_map

def generate_empty_latent(width=512, height=512, batch_size=1):
    return {"samples": torch.zeros([batch_size, 4, height // 8, width // 8], device=torch.device('cpu'))}

def clip_encode_single(clip, text, verbose=False):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens(tokens, return_pooled=True)

def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
            scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, callback_function=None, refiner=None, refiner_switch=-1,
            previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False):
    # This function is simplified for illustration
    return latent

def decode_vae(vae, latent_image, tiled=False):
    return latent_image["samples"]

def generate_image(prompt, steps=30, cfg_scale=7.0, width=512, height=512, sampler_name='dpmpp_2m_sde_gpu', scheduler_name='karras', seed=42):
    # Initialize necessary settings
    pid = os.getpid()
    patch_settings[pid] = PatchSettings()
    
    # Set environment variables
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Load models
    base_model_path = 'path_to_base_model'
    vae_model_path = 'path_to_vae_model'
    unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(base_model_path, vae_filename_param=vae_model_path)
    model = StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=base_model_path, vae_filename=vae_model_path)
    
    # Encode the prompt
    cond = clip_encode_single(model.clip_with_lora, prompt)
    positive_cond = [[cond[0], {"pooled_output": cond[1]}]]
    negative_cond = [[cond[0], {"pooled_output": cond[1]}]]  # Example negative prompt, usually a generic negative prompt is used
    
    # Generate empty latent
    latent = generate_empty_latent(width=width, height=height)
    
    # Sample the image
    sampled_latent = ksampler(
        model=model.unet_with_lora,
        positive=positive_cond,
        negative=negative_cond,
        latent=latent,
        steps=steps,
        cfg=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler_name,
        seed=seed
    )
    
    # Decode the image from latent
    decoded_image = decode_vae(vae=model.vae, latent_image=sampled_latent)
    
    # Convert the image to base64 (or save it as a file)
    base64_image = narray_to_base64img(decoded_image.cpu().numpy())
    
    return base64_image

# Example call to the function
image_base64 = generate_image(prompt="A beautiful landscape with mountains and a river.", steps=50, cfg_scale=7.5, width=512, height=512)
print(image_base64)  # This will print the base64 string of the generated image
