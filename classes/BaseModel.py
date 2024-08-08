import torch
import ldm_patched.modules.model_management
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel, Timestep
from classes.ModelType import ModelType
from utils.FooocusUtils import FooocusUtils
import ldm_patched.modules.utils as utils

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
