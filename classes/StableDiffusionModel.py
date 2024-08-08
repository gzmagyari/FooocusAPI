import os
import torch
from modules.util import get_file_from_folder_list
from modules.lora import match_lora
import modules.config
import modules.patch
import ldm_patched.modules.model_management
from FooocusUtils import FooocusUtils

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
