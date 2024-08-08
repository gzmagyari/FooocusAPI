from enum import Enum
import torch
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.model_sampling import EPS, V_PREDICTION, ModelSamplingDiscrete, ModelSamplingContinuousEDM
import ldm_patched.modules.model_management

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

class ModelType(Enum):
        EPS = 1
        V_PREDICTION = 2
        V_PREDICTION_EDM = 3

class FooocusUtils:
    @staticmethod
    def model_lora_keys_unet(model, key_map={}):
        sdk = model.state_dict().keys()

        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k

        diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(model.model_config.unet_config)
        for k in diffusers_keys:
            if k.endswith(".weight"):
                unet_key = "diffusion_model.{}".format(diffusers_keys[k])
                key_lora = k[:-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = unet_key

                diffusers_lora_prefix = ["", "unet."]
                for p in diffusers_lora_prefix:
                    diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                    if diffusers_lora_key.endswith(".to_out.0"):
                        diffusers_lora_key = diffusers_lora_key[:-2]
                    key_map[diffusers_lora_key] = unet_key
        return key_map

    @staticmethod
    def model_lora_keys_clip(model, key_map={}):
        sdk = model.state_dict().keys()

        text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
        clip_l_present = False
        for b in range(32): #TODO: clean up
            for c in LORA_CLIP_MAP:
                k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
                if k in sdk:
                    lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                    key_map[lora_key] = k
                    lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k

                k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
                if k in sdk:
                    lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                    key_map[lora_key] = k
                    lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                    key_map[lora_key] = k
                    clip_l_present = True
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k

                k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
                if k in sdk:
                    if clip_l_present:
                        lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                        key_map[lora_key] = k
                        lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                        key_map[lora_key] = k
                    else:
                        lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #TODO: test if this is correct for SDXL-Refiner
                        key_map[lora_key] = k
                        lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                        key_map[lora_key] = k

        return key_map        

    @staticmethod
    def unclip_adm(unclip_conditioning, device, noise_augmentor, noise_augment_merge=0.0, seed=None):
        adm_inputs = []
        weights = []
        noise_aug = []
        for unclip_cond in unclip_conditioning:
            for adm_cond in unclip_cond["clip_vision_output"].image_embeds:
                weight = unclip_cond["strength"]
                noise_augment = unclip_cond["noise_augmentation"]
                noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
                c_adm, noise_level_emb = noise_augmentor(adm_cond.to(device), noise_level=torch.tensor([noise_level], device=device), seed=seed)
                adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
                weights.append(weight)
                noise_aug.append(noise_augment)
                adm_inputs.append(adm_out)

        if len(noise_aug) > 1:
            adm_out = torch.stack(adm_inputs).sum(0)
            noise_augment = noise_augment_merge
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
            c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], noise_level=torch.tensor([noise_level], device=device))
            adm_out = torch.cat((c_adm, noise_level_emb), 1)

        return adm_out


    @staticmethod
    def sdxl_pooled(args, noise_augmentor):
        if "unclip_conditioning" in args:
            return FooocusUtils.unclip_adm(args.get("unclip_conditioning", None), args["device"], noise_augmentor, seed=args.get("seed", 0) - 10)[:,:1280]
        else:
            return args["pooled_output"]

    @staticmethod
    def model_sampling(model_config, model_type):
        s = ModelSamplingDiscrete

        if model_type == ModelType.EPS:
            c = EPS
        elif model_type == ModelType.V_PREDICTION:
            c = V_PREDICTION
        elif model_type == ModelType.V_PREDICTION_EDM:
            c = V_PREDICTION
            s = ModelSamplingContinuousEDM

        class ModelSampling(s, c):
            pass

        return ModelSampling(model_config)

    @staticmethod
    def normal_scheduler(model, steps, sgm=False, floor=False):
        s = model.model_sampling
        start = s.timestep(s.sigma_max)
        end = s.timestep(s.sigma_min)

        if sgm:
            timesteps = torch.linspace(start, end, steps + 1)[:-1]
        else:
            timesteps = torch.linspace(start, end, steps)

        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(s.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    @staticmethod
    def simple_scheduler(model, steps):
        s = model.model_sampling
        sigs = []
        ss = len(s.sigmas) / steps
        for x in range(steps):
            sigs += [float(s.sigmas[-(1 + int(x * ss))])]
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    @staticmethod
    def ddim_scheduler(model, steps):
        s = model.model_sampling
        sigs = []
        ss = len(s.sigmas) // steps
        x = 1
        while x < len(s.sigmas):
            sigs += [float(s.sigmas[x])]
            x += ss
        sigs = sigs[::-1]
        sigs += [0.0]
        return torch.FloatTensor(sigs)


    @staticmethod
    def calculate_sigmas_scheduler(model, scheduler_name, steps):
        if scheduler_name == "karras":
            sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
        elif scheduler_name == "exponential":
            sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
        elif scheduler_name == "normal":
            sigmas = FooocusUtils.normal_scheduler(model, steps)
        elif scheduler_name == "simple":
            sigmas = FooocusUtils.simple_scheduler(model, steps)
        elif scheduler_name == "ddim_uniform":
            sigmas = FooocusUtils.ddim_scheduler(model, steps)
        elif scheduler_name == "sgm_uniform":
            sigmas = FooocusUtils.normal_scheduler(model, steps, sgm=True)
        else:
            print("error invalid scheduler", scheduler_name)
        return sigmas

    @staticmethod
    def sampler_object(name):
            #if name == "uni_pc":
                #sampler = UNIPC()
            #elif name == "uni_pc_bh2":
                #sampler = UNIPCBH2()
            if name == "ddim":
                sampler = ldm_patched.modules.samplers.ksampler("euler", inpaint_options={"random": True})
            else:
                sampler = ldm_patched.modules.samplers.ksampler(name)
            return sampler 
