import os
import ssl
import sys
import uuid
import datetime
import json
from enum import Enum
import asyncio
import traceback
import time
import shared
import random
import copy
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import math
from fastapi import (
    APIRouter, Header
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import threading
import re

from build_launcher import build_launcher
from modules.launch_util import delete_folder_content
from apis.models.requests import CommonRequest
from apis.utils.pre_process import pre_worker
from modules.async_worker import AsyncTask
from apis.utils.api_utils import params_to_params
from apis.utils.sql_client import GenerateRecord
from modules.config import path_outputs
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel, Timestep
from modules.util import get_file_from_folder_list
from modules.lora import match_lora
from ldm_patched.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
import ldm_patched.modules.utils as utils
import modules.config
import modules.patch
import ldm_patched.modules.model_management
from apis.models.response import RecordResponse
from apis.models.base import CurrentTask
from apis.utils.img_utils import (
    narray_to_base64img, base64_to_image
)
from apis.utils.post_worker import post_worker
from modules import config
from modules.patch import PatchSettings, patch_settings
import modules.flags as flags
import modules.inpaint_worker as inpaint_worker
import modules.constants as constants
import extras.ip_adapter as ip_adapter
import fooocus_version
import args_manager
import extras.face_crop as face_crop
from FooocusUtils import FooocusUtils
from extras.censor import default_censor
from modules.sdxl_styles import apply_style, get_random_style, fooocus_expansion, apply_arrays, random_style_name
from modules.private_logger import log
from extras.expansion import safe_str
from modules.util import (remove_empty_str, HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil,
                            get_shape_ceil, resample_image, erode_or_dilate, get_enabled_loras,
                            parse_lora_references_from_prompt, apply_wildcards)
from modules.upscaler import perform_upscale
from modules.flags import Performance
from modules.meta_parser import MetadataScheme
from modules.sample_hijack import clip_separate
import extras.vae_interpose as vae_interpose
from extras.expansion import FooocusExpansion
import modules.core as core
from ldm_patched.modules.sd import load_checkpoint_guess_config
from modules.config import path_embeddings
import extras.preprocessors as preprocessors

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

MAX_RESOLUTION=8192
SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "tcd", "edm_playground_v2.5"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

def ini_args():
    from args_manager import args
    return args

build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)

engine = create_engine(
    f"sqlite:///{path_outputs}/db.sqlite3",
    connect_args={"check_same_thread": False},
    future=True
)
Session = sessionmaker(bind=engine, autoflush=True)
session = Session()

class ModelType(Enum):
        EPS = 1
        V_PREDICTION = 2
        V_PREDICTION_EDM = 3

class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype

        if not unet_config.get("disable_unet_model_creation", False):
            if self.manual_cast_dtype is not None:
                operations = ldm_patched.modules.ops.manual_cast
            else:
                operations = ldm_patched.modules.ops.disable_weight_init
            self.diffusion_model = UNetModel(**unet_config, device=device, operations=operations)
        self.model_type = model_type
        self.model_sampling = FooocusUtils.model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0
        self.inpaint_model = False
        print("model_type", model_type.name)
        print("UNet ADM Dimension", self.adm_channels)

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        if self.inpaint_model:
            concat_keys = ("mask", "masked_image")
            cond_concat = []
            denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
            concat_latent_image = kwargs.get("concat_latent_image", None)
            if concat_latent_image is None:
                concat_latent_image = kwargs.get("latent_image", None)
            else:
                concat_latent_image = self.process_latent_in(concat_latent_image)

            noise = kwargs.get("noise", None)
            device = kwargs["device"]

            if concat_latent_image.shape[1:] != noise.shape[1:]:
                concat_latent_image = utils.common_upscale(concat_latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

            concat_latent_image = utils.resize_to_batch_size(concat_latent_image, noise.shape[0])

            if len(denoise_mask.shape) == len(noise.shape):
                denoise_mask = denoise_mask[:,:1]

            denoise_mask = denoise_mask.reshape((-1, 1, denoise_mask.shape[-2], denoise_mask.shape[-1]))
            if denoise_mask.shape[-2:] != noise.shape[-2:]:
                denoise_mask = utils.common_upscale(denoise_mask, noise.shape[-1], noise.shape[-2], "bilinear", "center")
            denoise_mask = utils.resize_to_batch_size(denoise_mask.round(), noise.shape[0])

            def blank_inpaint_image_like(latent_image):
                blank_image = torch.ones_like(latent_image)
                # these are the values for "zero" in pixel space translated to latent space
                blank_image[:,0] *= 0.8223
                blank_image[:,1] *= -0.6876
                blank_image[:,2] *= 0.6364
                blank_image[:,3] *= 0.1380
                return blank_image

            for ck in concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask.to(device))
                    elif ck == "masked_image":
                        cond_concat.append(concat_latent_image.to(device)) #NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:,:1])
                    elif ck == "masked_image":
                        cond_concat.append(blank_inpaint_image_like(noise))
            data = torch.cat(cond_concat, dim=1)
            out['c_concat'] = ldm_patched.modules.conds.CONDNoiseShape(data)

        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = ldm_patched.modules.conds.CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = ldm_patched.modules.conds.CONDCrossAttn(cross_attn)

        return out

class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]), )

class VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", ),
                            "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64})
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "_for_testing"

    def decode(self, vae, samples, tile_size):
        return (vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, ), )

class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

class KSamplerX0Inpaint(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + (self.latent_image + self.noise * sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))) * latent_mask
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

class SDXL(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = FooocusUtils.sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)

class SDXLRefiner(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = FooocusUtils.sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)

        if kwargs.get("prompt_type", "") == "negative":
            aesthetic_score = kwargs.get("aesthetic_score", 2.5)
        else:
            aesthetic_score = kwargs.get("aesthetic_score", 6)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class EmptyLatentImage:
    def __init__(self):
        self.device = ldm_patched.modules.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                            "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )
    
class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x
    
class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None, vae_filename=None):
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
            self.lora_key_map_unet = FooocusUtils.model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = FooocusUtils.model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})
            
    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras):
        assert isinstance(loras, list)

        if self.visited_loras == str(loras):
            return

        self.visited_loras = str(loras)

        if self.unet is None:
            return

        print(f'Request to load LoRAs {str(loras)} for model [{self.filename}].')

        loras_to_load = []

        for filename, weight in loras:
            if filename == 'None':
                continue

            if os.path.exists(filename):
                lora_filename = filename
            else:
                lora_filename = get_file_from_folder_list(filename, modules.config.paths_loras)

            if not os.path.exists(lora_filename):
                print(f'Lora file not found: {lora_filename}')
                continue

            loras_to_load.append((lora_filename, weight))

        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        for lora_filename, weight in loras_to_load:
            lora_unmatch = ldm_patched.modules.utils.load_torch_file(lora_filename, safe_load=False)
            lora_unet, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_unet)
            lora_clip, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_clip)

            if len(lora_unmatch) > 12:
                # model mismatch
                continue

            if len(lora_unmatch) > 0:
                print(f'Loaded LoRA [{lora_filename}] for model [{self.filename}] '
                    f'with unmatched keys {list(lora_unmatch.keys())}')

            if self.unet_with_lora is not None and len(lora_unet) > 0:
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, weight)
                print(f'Loaded LoRA [{lora_filename}] for UNet [{self.filename}] '
                    f'with {len(loaded_keys)} keys at weight {weight}.')
                for item in lora_unet:
                    if item not in loaded_keys:
                        print("UNet LoRA key skipped: ", item)

            if self.clip_with_lora is not None and len(lora_clip) > 0:
                loaded_keys = self.clip_with_lora.add_patches(lora_clip, weight)
                print(f'Loaded LoRA [{lora_filename}] for CLIP [{self.filename}] '
                    f'with {len(loaded_keys)} keys at weight {weight}.')
                for item in lora_clip:
                    if item not in loaded_keys:
                        print("CLIP LoRA key skipped: ", item)

class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = FooocusUtils.calculate_sigmas_scheduler(self.model, self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = FooocusUtils.sampler_object(self.sampler)

        return ldm_patched.modules.samplers.sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


router = APIRouter()

# @router.post(
#         path="/v1/engine/generate/",
#         summary="Generate endpoint all in one",
#         tags=["GenerateV1"])
# async def generate_routes(
#     common_request: CommonRequest,
#     accept: str = Header(None)):
#     try:
#         accept, ext = accept.lower().split("/")
#         if ext not in ["png", "jpg", "jpeg", "webp"]:
#             ext = 'png'
#     except ValueError:
#         pass

#     return await async_worker(request=common_request, wait_for_result=True)


# def run_server(arguments):
#     api_port = 8888
#     uvicorn.run(app, host=arguments.listen, port=api_port)


#run_server(args_manager.args)


class FooocusModel():
    def __init__(self):
        self.async_tasks = []
        self.loaded_ControlNets = {}
        self.VAE_approx_models = {}
    
        self.model_base = StableDiffusionModel()
        self.model_refiner = StableDiffusionModel()

        self.final_expansion = None
        self.final_unet = None
        self.final_clip = None
        self.final_vae = None
        self.final_refiner_unet = None
        self.final_refiner_vae = None

        self.opEmptyLatentImage = EmptyLatentImage()
        self.opVAEDecodeTiled = VAEDecodeTiled()
        self.opVAEDecode = VAEDecode()

        self.pid = os.getpid()
        print(f'Started worker with PID {self.pid}')

        try:
            async_gradio_app = shared.gradio_root
            flag = f'''App started successful.'''
            if async_gradio_app.share:
                flag += f''' or {async_gradio_app.share_url}'''
            print(flag)
        except Exception as e:
            print(e)

    def progressbar(self, async_task, number, text):
        print(f'[Fooocus] {text}')
        async_task.yields.append(['preview', (number, text, None)])

    def yield_result(self, async_task, imgs, black_out_nsfw, censor=True, do_not_show_finished_images=False,
                     progressbar_index=flags.preparation_step_count):
        if not isinstance(imgs, list):
            imgs = [imgs]

        if censor and (modules.config.default_black_out_nsfw or black_out_nsfw):
            self.progressbar(async_task, progressbar_index, 'Checking for NSFW content ...')
            imgs = default_censor(imgs)

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        async_task.yields.append(['results', async_task.results])
        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def load_model(self, ckpt_filename, vae_filename=None):
        unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(ckpt_filename, embedding_directory=path_embeddings,
                                                                    vae_filename_param=vae_filename)
        return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=ckpt_filename, vae_filename=vae_filename)

    
    @torch.no_grad()
    @torch.inference_mode()
    def refresh_base_model(self, name, vae_name=None):
        filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

        vae_filename = None
        if vae_name is not None and vae_name != modules.flags.default_vae:
            vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

        if model_base.filename == filename and model_base.vae_filename == vae_filename:
            return

        model_base = self.load_model(filename, vae_filename)
        print(f'Base model loaded: {model_base.filename}')
        print(f'VAE loaded: {model_base.vae_filename}')
        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def synthesize_refiner_model(self):
        print('Synthetic Refiner Activated')

        self.model_refiner = StableDiffusionModel(
            unet=self.model_base.unet,
            vae=self.model_base.vae,
            clip=self.model_base.clip,
            clip_vision=self.model_base.clip_vision,
            filename=self.model_base.filename
        )
        self.model_refiner.vae = None
        self.model_refiner.clip = None
        self.model_refiner.clip_vision = None

        return

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_refiner_model(self, name):
        filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)
        if model_refiner.filename == filename:
            return

        model_refiner = StableDiffusionModel()

        if name == 'None':
            print(f'Refiner unloaded.')
            return

        model_refiner = self.load_model(filename)
        print(f'Refiner model loaded: {model_refiner.filename}')

        if isinstance(model_refiner.unet.model, SDXL):
            model_refiner.clip = None
            model_refiner.vae = None
        elif isinstance(model_refiner.unet.model, SDXLRefiner):
            model_refiner.clip = None
            model_refiner.vae = None
        else:
            model_refiner.clip = None

        return

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras, base_model_additional_loras=None):
        if not isinstance(base_model_additional_loras, list):
            base_model_additional_loras = []

        self.model_base.refresh_loras(loras + base_model_additional_loras)
        self.model_refiner.refresh_loras(loras)

        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def assert_model_integrity(self):
        error_message = None

        if not isinstance(self.model_base.unet_with_lora.model, SDXL):
            error_message = 'You have selected base model other than SDXL. This is not supported yet.'

        #if error_message is not None:
        #    raise NotImplementedError(error_message)

        return True
    
    @torch.no_grad()
    @torch.inference_mode()
    def prepare_text_encoder(self, async_call=True):
        if async_call:
            # TODO: make sure that this is always called in an async way so that users cannot feel it.
            pass
        self.assert_model_integrity()
        ldm_patched.modules.model_management.load_models_gpu([self.final_clip.patcher, self.final_expansion.patcher])
        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def clear_all_caches(self):
        self.final_clip.fcs_cond_cache = {}

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_everything(self, refiner_model_name, base_model_name, loras,
                        base_model_additional_loras=None, use_synthetic_refiner=False, vae_name=None):

        final_unet = None
        final_clip = None
        final_vae = None
        final_refiner_unet = None
        final_refiner_vae = None

        if use_synthetic_refiner and refiner_model_name == 'None':
            print('Synthetic Refiner Activated')
            self.refresh_base_model(base_model_name, vae_name)
            self.synthesize_refiner_model()
        else:
            self.refresh_refiner_model(refiner_model_name)
            self.refresh_base_model(base_model_name, vae_name)

        self.refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
        self.assert_model_integrity()

        final_unet = self.model_base.unet_with_lora
        final_clip = self.model_base.clip_with_lora
        final_vae = self.model_base.vae

        final_refiner_unet = self.model_refiner.unet_with_lora
        final_refiner_vae = self.model_refiner.vae

        if final_expansion is None:
            final_expansion = FooocusExpansion()

        self.prepare_text_encoder(async_call=True)
        self.clear_all_caches()
        return

    @torch.no_grad()
    @torch.inference_mode()
    def calculate_sigmas_all(self, sampler, model, scheduler, steps):
        discard_penultimate_sigma = False
        if sampler in ['dpm_2', 'dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = FooocusUtils.calculate_sigmas_scheduler(model, scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    @torch.no_grad()
    @torch.inference_mode()
    def calculate_sigmas(self, sampler, model, scheduler, steps, denoise):
        if denoise is None or denoise > 0.9999:
            sigmas = self.calculate_sigmas_all(sampler, model, scheduler, steps)
        else:
            new_steps = int(steps / denoise)
            sigmas = self.calculate_sigmas_all(sampler, model, scheduler, new_steps)
            sigmas = sigmas[-(steps + 1):]
        return sigmas

    @torch.no_grad()
    @torch.inference_mode()
    def vae_parse(self, latent):
        if self.final_refiner_vae is None:
            return latent

        result = vae_interpose.parse(latent["samples"])
        return {'samples': result}

    @torch.no_grad()
    @torch.inference_mode()
    def generate_empty_latent(self, width=1024, height=1024, batch_size=1):
        return self.opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]

    @torch.no_grad()
    @torch.inference_mode()
    def get_previewer(self, model):

        from modules.config import path_vae_approx
        is_sdxl = isinstance(model.model.latent_format, ldm_patched.modules.latent_formats.SDXL)
        vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

        if vae_approx_filename in self.VAE_approx_models:
            VAE_approx_model = self.VAE_approx_models[vae_approx_filename]
        else:
            sd = torch.load(vae_approx_filename, map_location='cpu')
            VAE_approx_model = VAEApprox()
            VAE_approx_model.load_state_dict(sd)
            del sd
            VAE_approx_model.eval()

            if ldm_patched.modules.model_management.should_use_fp16():
                VAE_approx_model.half()
                VAE_approx_model.current_type = torch.float16
            else:
                VAE_approx_model.float()
                VAE_approx_model.current_type = torch.float32

            VAE_approx_model.to(ldm_patched.modules.model_management.get_torch_device())
            self.VAE_approx_models[vae_approx_filename] = VAE_approx_model

    def get_models_from_cond(self, cond, model_type):
        models = []
        for c in cond:
            if model_type in c:
                models += [c[model_type]]
        return models


    def get_additional_models(self, positive, negative, dtype):
        """loads additional models in positive and negative conditioning"""
        control_nets = set(self.get_models_from_cond(positive, "control") + self.get_models_from_cond(negative, "control"))

        inference_memory = 0
        control_models = []
        for m in control_nets:
            control_models += m.get_models()
            inference_memory += m.inference_memory_requirements(dtype)

        gligen = self.get_models_from_cond(positive, "gligen") + self.get_models_from_cond(negative, "gligen")
        gligen = [x[1] for x in gligen]
        models = control_models + gligen
        return models, inference_memory


    def convert_cond(self, cond):
        out = []
        for c in cond:
            temp = c[1].copy()
            model_conds = temp.get("model_conds", {})
            if c[0] is not None:
                model_conds["c_crossattn"] = ldm_patched.modules.conds.CONDCrossAttn(c[0]) #TODO: remove
                temp["cross_attn"] = c[0]
            temp["model_conds"] = model_conds
            out.append(temp)
        return out

    def prepare_mask(self, noise_mask, shape, device):
        """ensures noise mask is of proper dimensions"""
        noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
        noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
        noise_mask = ldm_patched.modules.utils.repeat_to_batch_size(noise_mask, shape[0])
        noise_mask = noise_mask.to(device)
        return noise_mask


    def prepare_sampling(self, model, noise_shape, positive, negative, noise_mask):
        device = model.load_device
        positive = self.convert_cond(positive)
        negative = self.convert_cond(negative)

        if noise_mask is not None:
            noise_mask = self.prepare_mask(noise_mask, noise_shape, device)

        real_model = None
        models, inference_memory = self.get_additional_models(positive, negative, model.model_dtype())
        ldm_patched.modules.model_management.load_models_gpu([model] + models, model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:])) + inference_memory)
        real_model = model.model

        return real_model, positive, negative, noise_mask, models
    
    def cleanup_additional_models(self, models):
        """cleanup additional models that were loaded"""
        for m in models:
            if hasattr(m, 'cleanup'):
                m.cleanup()


    def sample(self, model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        real_model, positive_copy, negative_copy, noise_mask, models = self.prepare_sampling(model, noise.shape, positive, negative, noise_mask)

        noise = noise.to(model.load_device)
        latent_image = latent_image.to(model.load_device)

        sampler = KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

        samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
        samples = samples.to(ldm_patched.modules.model_management.intermediate_device())

        self.cleanup_additional_models(models)
        self.cleanup_additional_models(set(self.get_models_from_cond(positive_copy, "control") + self.get_models_from_cond(negative_copy, "control")))
        return samples


    @torch.no_grad()
    @torch.inference_mode()
    def ksampler(self, model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
                scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=False, callback_function=None, refiner=None, refiner_switch=-1,
                previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False):

        if sigmas is not None:
            sigmas = sigmas.clone().to(ldm_patched.modules.model_management.get_torch_device())

        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = ldm_patched.modules.sample.prepare_noise(latent_image, seed, batch_inds)

        if isinstance(noise_mean, torch.Tensor):
            noise = noise + noise_mean - torch.mean(noise, dim=1, keepdim=True)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        previewer = self.get_previewer(model)

        if previewer_start is None:
            previewer_start = 0

        if previewer_end is None:
            previewer_end = steps

        def callback(step, x0, x, total_steps):
            ldm_patched.modules.model_management.throw_exception_if_processing_interrupted()
            y = None
            if previewer is not None and not disable_preview:
                y = previewer(x0, previewer_start + step, previewer_end)
            if callback_function is not None:
                callback_function(previewer_start + step, x0, x, previewer_end, y)

        disable_pbar = False
        modules.sample_hijack.current_refiner = refiner
        modules.sample_hijack.refiner_switch_step = refiner_switch
        ldm_patched.modules.samplers.sample = modules.sample_hijack.sample_hacked

        try:
            samples = self.sample(model,
                                                        noise, steps, cfg, sampler_name, scheduler,
                                                        positive, negative, latent_image,
                                                        denoise=denoise, disable_noise=disable_noise,
                                                        start_step=start_step,
                                                        last_step=last_step,
                                                        force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                                        callback=callback,
                                                        disable_pbar=disable_pbar, seed=seed, sigmas=sigmas)

            out = latent.copy()
            out["samples"] = samples
        finally:
            modules.sample_hijack.current_refiner = None

        return out
    
    

    @torch.no_grad()
    @torch.inference_mode()
    def decode_vae(self, vae, latent_image, tiled=False):
        if tiled:
            return self.opVAEDecodeTiled.decode(samples=latent_image, vae=vae, tile_size=512)[0]
        else:
            return self.opVAEDecode.decode(samples=latent_image, vae=vae)[0]


    @torch.no_grad()
    @torch.inference_mode()
    def process_diffusion(self, positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint', disable_preview=False):
        target_unet, target_vae, target_refiner_unet, target_refiner_vae, target_clip \
            = self.final_unet, self.final_vae, self.final_refiner_unet, self.final_refiner_vae, self.final_clip

        assert refiner_swap_method in ['joint', 'separate', 'vae']

        if self.final_refiner_vae is not None and self.final_refiner_unet is not None:
            # Refiner Use Different VAE (then it is SD15)
            if denoise > 0.9:
                refiner_swap_method = 'vae'
            else:
                refiner_swap_method = 'joint'
                if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                    target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                        = self.final_unet, self.final_vae, None, None
                    print(f'[Sampler] only use Base because of partial denoise.')
                else:
                    positive_cond = clip_separate(positive_cond, target_model=self.final_refiner_unet.model, target_clip=self.final_clip)
                    negative_cond = clip_separate(negative_cond, target_model=self.final_refiner_unet.model, target_clip=self.final_clip)
                    target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                        = self.final_refiner_unet, self.final_refiner_vae, None, None
                    print(f'[Sampler] only use Refiner because of partial denoise.')

        print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

        if latent is None:
            initial_latent = self.generate_empty_latent(width=width, height=height, batch_size=1)
        else:
            initial_latent = latent

        minmax_sigmas = self.calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=self.final_unet.model, steps=steps, denoise=denoise)
        sigma_min, sigma_max = minmax_sigmas[minmax_sigmas > 0].min(), minmax_sigmas.max()
        sigma_min = float(sigma_min.cpu().numpy())
        sigma_max = float(sigma_max.cpu().numpy())
        print(f'[Sampler] sigma_min = {sigma_min}, sigma_max = {sigma_max}')

        modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
            initial_latent['samples'].to(ldm_patched.modules.model_management.get_torch_device()),
            sigma_min, sigma_max, seed=image_seed, cpu=False)

        decoded_latent = None

        if refiner_swap_method == 'joint':
            sampled_latent = self.ksampler(
                model=target_unet,
                refiner=target_refiner_unet,
                positive=positive_cond,
                negative=negative_cond,
                latent=initial_latent,
                steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                refiner_switch=switch,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview
            )
            decoded_latent = self.decode_vae(vae=target_vae, latent_image=sampled_latent, tiled=tiled)

        if refiner_swap_method == 'separate':
            sampled_latent = self.ksampler(
                model=target_unet,
                positive=positive_cond,
                negative=negative_cond,
                latent=initial_latent,
                steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=False,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview
            )
            print('Refiner swapped by changing ksampler. Noise preserved.')

            target_model = target_refiner_unet
            if target_model is None:
                target_model = target_unet
                print('Use base model to refine itself - this may because of developer mode.')

            sampled_latent = self.ksampler(
                model=target_model,
                positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
                negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
                latent=sampled_latent,
                steps=steps, start_step=switch, last_step=steps, disable_noise=True, force_full_denoise=True,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=switch,
                previewer_end=steps,
                disable_preview=disable_preview
            )

            target_model = target_refiner_vae
            if target_model is None:
                target_model = target_vae
            decoded_latent = self.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

        if refiner_swap_method == 'vae':
            modules.patch.patch_settings[os.getpid()].eps_record = 'vae'

            if modules.inpaint_worker.current_task is not None:
                modules.inpaint_worker.current_task.unswap()

            sampled_latent = self.ksampler(
                model=target_unet,
                positive=positive_cond,
                negative=negative_cond,
                latent=initial_latent,
                steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=True,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview
            )
            print('Fooocus VAE-based swap.')

            target_model = target_refiner_unet
            if target_model is None:
                target_model = target_unet
                print('Use base model to refine itself - this may because of developer mode.')

            sampled_latent = self.vae_parse(sampled_latent)

            k_sigmas = 1.4
            sigmas = self.calculate_sigmas(sampler=sampler_name,
                                    scheduler=scheduler_name,
                                    model=target_model.model,
                                    steps=steps,
                                    denoise=denoise)[switch:] * k_sigmas
            len_sigmas = len(sigmas) - 1

            noise_mean = torch.mean(modules.patch.patch_settings[os.getpid()].eps_record, dim=1, keepdim=True)

            if modules.inpaint_worker.current_task is not None:
                modules.inpaint_worker.current_task.swap()

            sampled_latent = self.ksampler(
                model=target_model,
                positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
                negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
                latent=sampled_latent,
                steps=len_sigmas, start_step=0, last_step=len_sigmas, disable_noise=False, force_full_denoise=True,
                seed=image_seed+1,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=switch,
                previewer_end=steps,
                sigmas=sigmas,
                noise_mean=noise_mean,
                disable_preview=disable_preview
            )

            target_model = target_refiner_vae
            if target_model is None:
                target_model = target_vae
            decoded_latent = self.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

        images = core.pytorch_to_numpy(decoded_latent)
        modules.patch.patch_settings[os.getpid()].eps_record = None
        return images

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_controlnets(self, model_paths):
        cache = {}
        for p in model_paths:
            if p is not None:
                if p in self.loaded_ControlNets:
                    cache[p] = self.loaded_ControlNets[p]
                else:
                    cache[p] = core.load_controlnet(p)
        self.loaded_ControlNets = cache
        return

    @torch.no_grad()
    @torch.inference_mode()
    def set_clip_skip(self, clip_skip: int):
        if self.final_clip is None:
            return

        self.final_clip.clip_layer(-abs(clip_skip))
        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def clip_encode_single(self, clip, text, verbose=False):
        cached = clip.fcs_cond_cache.get(text, None)
        if cached is not None:
            if verbose:
                print(f'[CLIP Cached] {text}')
            return cached
        tokens = clip.tokenize(text)
        result = clip.encode_from_tokens(tokens, return_pooled=True)
        clip.fcs_cond_cache[text] = result
        if verbose:
            print(f'[CLIP Encoded] {text}')
        return result
    
    @torch.no_grad()
    @torch.inference_mode()
    def clip_encode(self, texts, pool_top_k=1):
        if self.final_clip is None:
            return None
        if not isinstance(texts, list):
            return None
        if len(texts) == 0:
            return None

        cond_list = []
        pooled_acc = 0

        for i, text in enumerate(texts):
            cond, pooled = self.clip_encode_single(self.final_clip, text)
            cond_list.append(cond)
            if i < pool_top_k:
                pooled_acc += pooled

        return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]

    @torch.no_grad()
    @torch.inference_mode()
    def get_candidate_vae(self, steps, switch, denoise=1.0, refiner_swap_method='joint'):
        assert refiner_swap_method in ['joint', 'separate', 'vae']

        if self.final_refiner_vae is not None and self.final_refiner_unet is not None:
            if denoise > 0.9:
                return self.final_vae, self.final_refiner_vae
            else:
                if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                    return self.final_vae, None
                else:
                    return self.final_refiner_vae, None

        return self.final_vae, self.final_refiner_vae

    @torch.no_grad()
    @torch.inference_mode()
    def clone_cond(self, conds):
        results = []

        for c, p in conds:
            p = p["pooled_output"]

            if isinstance(c, torch.Tensor):
                c = c.clone()

            if isinstance(p, torch.Tensor):
                p = p.clone()

            results.append([c, {"pooled_output": p}])

        return results

    @torch.no_grad()
    @torch.inference_mode()
    def handler(self, async_task):
        execution_start_time = time.perf_counter()
        async_task.processing = True

        args = async_task.args
        args.reverse()

        prompt = args.pop()
        negative_prompt = args.pop()
        style_selections = args.pop()
        performance_selection = Performance(args.pop())
        aspect_ratios_selection = args.pop()
        image_number = args.pop()
        output_format = args.pop()
        image_seed = args.pop()
        read_wildcards_in_order = args.pop()
        sharpness = args.pop()
        guidance_scale = args.pop()
        base_model_name = args.pop()
        refiner_model_name = args.pop()
        refiner_switch = args.pop()
        loras = get_enabled_loras([(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in
                                   range(modules.config.default_max_lora_number)])
        input_image_checkbox = args.pop()
        current_tab = args.pop()
        uov_method = args.pop()
        uov_input_image = args.pop()
        outpaint_selections = args.pop()
        inpaint_input_image = args.pop()
        inpaint_additional_prompt = args.pop()
        inpaint_mask_image_upload = args.pop()

        disable_preview = args.pop()
        disable_intermediate_results = args.pop()
        disable_seed_increment = args.pop()
        black_out_nsfw = args.pop()
        adm_scaler_positive = args.pop()
        adm_scaler_negative = args.pop()
        adm_scaler_end = args.pop()
        adaptive_cfg = args.pop()
        clip_skip = args.pop()
        sampler_name = args.pop()
        scheduler_name = args.pop()
        vae_name = args.pop()
        overwrite_step = args.pop()
        overwrite_switch = args.pop()
        overwrite_width = args.pop()
        overwrite_height = args.pop()
        overwrite_vary_strength = args.pop()
        overwrite_upscale_strength = args.pop()
        mixing_image_prompt_and_vary_upscale = args.pop()
        mixing_image_prompt_and_inpaint = args.pop()
        debugging_cn_preprocessor = args.pop()
        skipping_cn_preprocessor = args.pop()
        canny_low_threshold = args.pop()
        canny_high_threshold = args.pop()
        refiner_swap_method = args.pop()
        controlnet_softness = args.pop()
        freeu_enabled = args.pop()
        freeu_b1 = args.pop()
        freeu_b2 = args.pop()
        freeu_s1 = args.pop()
        freeu_s2 = args.pop()
        debugging_inpaint_preprocessor = args.pop()
        inpaint_disable_initial_latent = args.pop()
        inpaint_engine = args.pop()
        inpaint_strength = args.pop()
        inpaint_respective_field = args.pop()
        inpaint_mask_upload_checkbox = args.pop()
        invert_mask_checkbox = args.pop()
        inpaint_erode_or_dilate = args.pop()

        save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
        metadata_scheme = MetadataScheme(
            args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        cn_tasks = {x: [] for x in flags.ip_list}
        for _ in range(flags.controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])
        try:
            outpaint_distance = args.pop()
            dt_left, dt_top, dt_right, dt_bottom = outpaint_distance[:4]
        except Exception:
            dt_left, dt_top, dt_right, dt_bottom = 0, 0, 0, 0

        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        if base_model_name == refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        steps = performance_selection.steps()

        performance_loras = []

        if performance_selection == Performance.EXTREME_SPEED:
            print('Enter LCM mode.')
            self.progressbar(async_task, 1, 'Downloading LCM components ...')
            performance_loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in LCM mode.')

            refiner_model_name = 'None'
            sampler_name = 'lcm'
            scheduler_name = 'lcm'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        elif performance_selection == Performance.LIGHTNING:
            print('Enter Lightning mode.')
            self.progressbar(async_task, 1, 'Downloading Lightning components ...')
            performance_loras += [(modules.config.downloading_sdxl_lightning_lora(), 1.0)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in Lightning mode.')

            refiner_model_name = 'None'
            sampler_name = 'euler'
            scheduler_name = 'sgm_uniform'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        elif performance_selection == Performance.HYPER_SD:
            print('Enter Hyper-SD mode.')
            self.progressbar(async_task, 1, 'Downloading Hyper-SD components ...')
            performance_loras += [(modules.config.downloading_sdxl_hyper_sd_lora(), 0.8)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in Hyper-SD mode.')

            refiner_model_name = 'None'
            sampler_name = 'dpmpp_sde_gpu'
            scheduler_name = 'karras'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        print(f'[Parameters] Adaptive CFG = {adaptive_cfg}')
        print(f'[Parameters] CLIP Skip = {clip_skip}')
        print(f'[Parameters] Sharpness = {sharpness}')
        print(f'[Parameters] ControlNet Softness = {controlnet_softness}')
        print(f'[Parameters] ADM Scale = '
              f'{adm_scaler_positive} : '
              f'{adm_scaler_negative} : '
              f'{adm_scaler_end}')

        patch_settings[self.pid] = PatchSettings(
            sharpness,
            adm_scaler_end,
            adm_scaler_positive,
            adm_scaler_negative,
            controlnet_softness,
            adaptive_cfg
        )

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)
        print(f'[Parameters] Seed = {seed}')

        goals = []
        tasks = []

        if input_image_checkbox:
            if (current_tab == 'uov' or (
                    current_tab == 'ip' and mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = performance_selection.steps_uov()

                    self.progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (current_tab == 'inpaint' or (
                    current_tab == 'ip' and mixing_image_prompt_and_inpaint)) \
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]

                if inpaint_mask_upload_checkbox:
                    if isinstance(inpaint_mask_image_upload, np.ndarray):
                        if inpaint_mask_image_upload.ndim == 3:
                            H, W, C = inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            inpaint_mask = np.maximum(inpaint_mask, inpaint_mask_image_upload)

                if int(inpaint_erode_or_dilate) != 0:
                    inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)

                if invert_mask_checkbox:
                    inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    self.progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
                    if inpaint_parameterized:
                        self.progressbar(async_task, 1, 'Downloading inpainter ...')
                        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.8
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if inpaint_additional_prompt != '':
                        if prompt == '':
                            prompt = inpaint_additional_prompt
                        else:
                            prompt = inpaint_additional_prompt + '\n' + prompt
                    goals.append('inpaint')
            if current_tab == 'ip' or \
                    mixing_image_prompt_and_vary_upscale or \
                    mixing_image_prompt_and_inpaint:
                goals.append('cn')
                self.progressbar(async_task, 1, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(cn_tasks[flags.cn_ip_face]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                self.progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        self.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        if overwrite_step > 0:
            steps = overwrite_step

        switch = int(round(steps * refiner_switch))

        if overwrite_switch > 0:
            switch = overwrite_switch

        if overwrite_width > 0:
            width = overwrite_width

        if overwrite_height > 0:
            height = overwrite_height

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        self.progressbar(async_task, 1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            self.progressbar(async_task, 2, 'Loading models ...')

            lora_filenames = modules.util.remove_performance_lora(modules.config.lora_filenames, performance_selection)
            loras, prompt = parse_lora_references_from_prompt(prompt, loras, modules.config.default_max_lora_number, lora_filenames=lora_filenames)
            loras += performance_loras

            self.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner, vae_name=vae_name)

            self.set_clip_skip(clip_skip)

            self.progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []

            for i in range(image_number):
                if disable_seed_increment:
                    task_seed = seed % (constants.MAX_SEED + 1)
                else:
                    task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
                task_prompt = apply_wildcards(prompt, task_rng, i, read_wildcards_in_order)
                task_prompt = apply_arrays(task_prompt, i)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, read_wildcards_in_order)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in
                                               extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in
                                               extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                task_styles = style_selections.copy()
                if use_style:
                    for i, s in enumerate(task_styles):
                        if s == random_style_name:
                            s = get_random_style(task_rng)
                            task_styles[i] = s
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                    styles=task_styles
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    self.progressbar(async_task, 4, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = self.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                self.progressbar(async_task, 5, f'Encoding positive #{i + 1} ...')
                t['c'] = self.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = self.clone_cond(t['c'])
                else:
                    self.progressbar(async_task, 6, f'Encoding negative #{i + 1} ...')
                    t['uc'] = self.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            self.progressbar(async_task, 7, 'Image processing ...')

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if overwrite_vary_strength > 0:
                denoising_strength = overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            self.progressbar(async_task, 8, 'VAE encoding ...')

            candidate_vae, _ = self.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            self.progressbar(async_task, 9, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            print(f'Image upscaled.')

            if '1.5x' in uov_method:
                f = 1.5
            elif '2x' in uov_method:
                f = 2.0
            else:
                # @freek99
                pattern = r"([0-9]+(?:\.[0-9]+)?)x"
                matches = re.findall(pattern, uov_method)
                try:
                    f = float(matches[0])
                    if f < 1.0:
                        f = 1.0
                    if f > 5.0:
                        f = 5.0
                except Exception:
                    f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                if modules.config.default_black_out_nsfw or black_out_nsfw:
                    self.progressbar(async_task, 100, 'Checking for NSFW content ...')
                    uov_input_image = default_censor(uov_input_image)
                self.progressbar(async_task, 100, 'Saving image to system ...')
                uov_input_image_path = log(uov_input_image, d, output_format=output_format)
                self.yield_result(async_task, uov_input_image_path, black_out_nsfw, False, do_not_show_finished_images=True)
                return

            tiled = True
            denoising_strength = 0.382

            if overwrite_upscale_strength > 0:
                denoising_strength = overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            self.progressbar(async_task, 10, 'VAE encoding ...')

            candidate_vae, _ = self.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    distance_top = int(H * 0.3)
                    if dt_top > 0:
                        distance_top = dt_top

                    inpaint_image = np.pad(inpaint_image, [[distance_top, 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[distance_top, 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if 'bottom' in outpaint_selections:
                    distance_bottom = int(H * 0.3)
                    if dt_bottom > 0:
                        distance_bottom = dt_bottom

                    inpaint_image = np.pad(inpaint_image, [[0, distance_bottom], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, distance_bottom], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    distance_left = int(W * 0.3)
                    if dt_left > 0:
                        distance_left = dt_left

                    inpaint_image = np.pad(inpaint_image, [[0, 0], [distance_left, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [distance_left, 0]], mode='constant',
                                          constant_values=255)
                if 'right' in outpaint_selections:
                    distance_right = int(W * 0.3)
                    if dt_right > 0:
                        distance_right = dt_right

                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, distance_right], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, distance_right]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                inpaint_strength = 1.0
                inpaint_respective_field = 1.0

            denoising_strength = inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            if debugging_inpaint_preprocessor:
                self.yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(), black_out_nsfw,
                             do_not_show_finished_images=True)
                return

            self.progressbar(async_task, 11, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = self.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                self.progressbar(async_task, 12, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            self.progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=final_unet
                )

            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img, canny_low_threshold, canny_high_threshold)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, black_out_nsfw, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, black_out_nsfw, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, black_out_nsfw, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not skipping_cn_preprocessor:
                    cn_img = face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, black_out_nsfw, do_not_show_finished_images=True)
                    return

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                final_unet = ip_adapter.patch_model(final_unet, all_ip_tasks)

        if freeu_enabled:
            print(f'FreeU is enabled!')
            final_unet = core.apply_freeu(
                final_unet,
                freeu_b1,
                freeu_b2,
                freeu_s1,
                freeu_s2
            )

        all_steps = steps * image_number

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name in ['lcm', 'tcd']:
            final_scheduler_name = 'sgm_uniform'

            def patch_discrete(unet):
                return core.opModelSamplingDiscrete.patch(
                    final_unet,
                    sampling=scheduler_name,
                    zsnr=False)[0]

            if final_unet is not None:
                final_unet = patch_discrete(final_unet)
            if final_refiner_unet is not None:
                final_refiner_unet = patch_discrete(final_refiner_unet)
            print(f'Using {scheduler_name} scheduler.')
        elif scheduler_name == 'edm_playground_v2.5':
            final_scheduler_name = 'karras'

            def patch_edm(unet):
                return core.opModelSamplingContinuousEDM.patch(
                    unet,
                    sampling=scheduler_name,
                    sigma_max=120.0,
                    sigma_min=0.002)[0]

            if final_unet is not None:
                final_unet = patch_edm(final_unet)
            if final_refiner_unet is not None:
                final_refiner_unet = patch_edm(final_refiner_unet)

            print(f'Using {scheduler_name} scheduler.')

        async_task.yields.append(['preview', (flags.preparation_step_count, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            async_task.yields.append(['preview', (
                int(flags.preparation_step_count + (100 - flags.preparation_step_count) * float(done_steps) / float(all_steps)),
                f'Sampling step {step + 1}/{total_steps}, image {current_task_id + 1}/{image_number} ...', y)])

        for current_task_id, task in enumerate(tasks):
            current_progress = int(flags.preparation_step_count + (100 - flags.preparation_step_count) * float(current_task_id * steps) / float(all_steps))
            self.progressbar(async_task, current_progress, f'Preparing task {current_task_id + 1}/{image_number} ...')
            execution_start_time = time.perf_counter()

            try:
                if async_task.last_stop is not False:
                    ldm_patched.modules.model_management.interrupt_current_processing()
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                self.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = self.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method,
                    disable_preview=disable_preview
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                img_paths = []
                current_progress = int(flags.preparation_step_count + (100 - flags.preparation_step_count) * float((current_task_id + 1) * steps) / float(all_steps))
                if modules.config.default_black_out_nsfw or black_out_nsfw:
                    self.progressbar(async_task, current_progress, 'Checking for NSFW content ...')
                    imgs = default_censor(imgs)

                self.progressbar(async_task, current_progress, f'Saving image {current_task_id + 1}/{image_number} to system ...')
                for x in imgs:
                    d = [('Prompt', 'prompt', task['log_positive_prompt']),
                         ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                         ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                         ('Styles', 'styles',
                          str(task['styles'] if not use_expansion else [fooocus_expansion] + task['styles'])),
                         ('Performance', 'performance', performance_selection.value)]

                    if performance_selection.steps() != steps:
                        d.append(('Steps', 'steps', steps))

                    d += [('Resolution', 'resolution', str((width, height))),
                          ('Guidance Scale', 'guidance_scale', guidance_scale),
                          ('Sharpness', 'sharpness', sharpness),
                          ('ADM Guidance', 'adm_guidance', str((
                              modules.patch.patch_settings[self.pid].positive_adm_scale,
                              modules.patch.patch_settings[self.pid].negative_adm_scale,
                              modules.patch.patch_settings[self.pid].adm_scaler_end))),
                          ('Base Model', 'base_model', base_model_name),
                          ('Refiner Model', 'refiner_model', refiner_model_name),
                          ('Refiner Switch', 'refiner_switch', refiner_switch)]

                    if refiner_model_name != 'None':
                        if overwrite_switch > 0:
                            d.append(('Overwrite Switch', 'overwrite_switch', overwrite_switch))
                        if refiner_swap_method != flags.refiner_swap_method:
                            d.append(('Refiner Swap Method', 'refiner_swap_method', refiner_swap_method))
                    if modules.patch.patch_settings[self.pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                        d.append(
                            ('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[self.pid].adaptive_cfg))

                    if clip_skip > 1:
                        d.append(('CLIP Skip', 'clip_skip', clip_skip))
                    d.append(('Sampler', 'sampler', sampler_name))
                    d.append(('Scheduler', 'scheduler', scheduler_name))
                    d.append(('VAE', 'vae', vae_name))
                    d.append(('Seed', 'seed', str(task['task_seed'])))

                    if freeu_enabled:
                        d.append(('FreeU', 'freeu', str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))))

                    for li, (n, w) in enumerate(loras):
                        if n != 'None':
                            d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

                    metadata_parser = None
                    if save_metadata_to_images:
                        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                        metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                                 task['log_negative_prompt'], task['negative'],
                                                 steps, base_model_name, refiner_model_name, loras, vae_name)
                    d.append(('Metadata Scheme', 'metadata_scheme',
                              metadata_scheme.value if save_metadata_to_images else save_metadata_to_images))
                    d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))
                    img_paths.append(log(x, d, metadata_parser, output_format, task))

                self.yield_result(async_task, img_paths, black_out_nsfw, False,
                             do_not_show_finished_images=len(tasks) == 1 or disable_intermediate_results)
            except ldm_patched.modules.model_management.InterruptProcessingException as e:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')
        async_task.processing = False
        return

    def start(self):
        while True:
            time.sleep(0.01)
            if len(self.async_tasks) > 0:
                print("Processing task");
                task = self.async_tasks.pop(0)
                generate_image_grid = task.args.pop(0)

                try:
                    self.handler(task)
                    task.yields.append(['finish', task.results])
                    self.prepare_text_encoder(async_call=True)
                except:
                    traceback.print_exc()
                    task.yields.append(['finish', task.results])
                finally:
                    if self.pid in modules.patch.patch_settings:
                        del modules.patch.patch_settings[self.pid]
        pass

    async def startInBackground(self):
        threading.Thread(target=self.start, daemon=True).start()

    async def execute_in_background(self, task: AsyncTask, raw_req: CommonRequest, in_queue_mills):
        finished = False
        started = False
        while not finished:
            await asyncio.sleep(0.2)
            if len(task.yields) > 0:
                if not started:
                    started = True
                    started_at = int(datetime.datetime.now().timestamp() * 1000)
                    CurrentTask.ct = RecordResponse(
                        task_id=task.task_id,
                        req_params=json.loads(raw_req.model_dump_json()),
                        in_queue_mills=in_queue_mills,
                        start_mills=started_at,
                        task_status="running",
                        progress=0
                    )
                    CurrentTask.task = task
                flag, product = task.yields.pop(0)
                if flag == 'preview':
                    if len(task.yields) > 0:
                        if task.yields[0][0] == 'preview':
                            continue
                    percentage, _, image = product
                    CurrentTask.ct.progress = percentage
                    CurrentTask.ct.preview = narray_to_base64img(image)
                if flag == 'finish':
                    finished = True
                    CurrentTask.task = None
                    return await post_worker(task=task, started_at=started_at)

    async def async_worker(self, request: CommonRequest, wait_for_result: bool = False) -> dict:
        if request.webhook_url is None or request.webhook_url == "":
            request.webhook_url = os.environ.get("WEBHOOK_URL")
        raw_req, request = await pre_worker(request)
        task_id = uuid.uuid4().hex
        task = AsyncTask(
            task_id=task_id,
            args=params_to_params(request)
        )
        self.async_tasks.append(task)
        in_queue_mills = int(datetime.datetime.now().timestamp() * 1000)
        session.add(GenerateRecord(
            task_id=task.task_id,
            req_params=json.loads(raw_req.model_dump_json()),
            webhook_url=raw_req.webhook_url,
            in_queue_mills=in_queue_mills
        ))
        session.commit()

        if wait_for_result:
            res = await self.execute_in_background(task, raw_req, in_queue_mills)
            return json.loads(res)

        asyncio.create_task(self.execute_in_background(task, raw_req, in_queue_mills))
        return RecordResponse(task_id=task_id, task_status="pending").model_dump()

model = FooocusModel()
model.startInBackground()

request = CommonRequest(
    prompt="a cute cat, crisp clear, 4k, vivid colors, high resolution",
    negative_prompt="blurry, low resolution, pixelated",
    performance_selection = "Quality"
)

async def main():
    result = await model.async_worker(request=request, wait_for_result=True)
    return result

result = asyncio.run(main())
#writing the result to a file
with open('result.txt', 'w') as f:
    #serializing
    f.write(json.dumps(result))
base64str = result["base64_result"][0]
base64_to_image(base64str, "./result.png")
print("Image saved as result.png")
