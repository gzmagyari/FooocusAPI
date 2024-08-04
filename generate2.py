import torch
import modules.config
from modules.util import get_file_from_folder_list
from ldm_patched.modules.model_management import load_models_gpu, get_torch_device, intermediate_device
from modules.core import pytorch_to_numpy
from apis.utils.img_utils import narray_to_base64img

# ... (Other imports and global variables like model_base, final_unet, final_vae, etc.)

@torch.no_grad()
@torch.inference_mode()
def generate_single_image(prompt, negative_prompt="", steps=30, cfg_scale=7.0, sampler_name='dpmpp_2m_sde_gpu', scheduler_name='karras', seed=None, width=512, height=512):
    """
    Generates a single image from a prompt.

    Args:
        prompt (str): The text prompt describing the desired image.
        negative_prompt (str, optional): The text prompt describing what to avoid in the image. Defaults to "".
        steps (int, optional): The number of sampling steps. Defaults to 30.
        cfg_scale (float, optional): The classifier-free guidance scale. Defaults to 7.0.
        sampler_name (str, optional): The name of the sampler to use. Defaults to 'dpmpp_2m_sde_gpu'.
        scheduler_name (str, optional): The name of the scheduler to use. Defaults to 'karras'.
        seed (int, optional): The random seed for image generation. Defaults to None.
        width (int, optional): The width of the image. Defaults to 512.
        height (int, optional): The height of the image. Defaults to 512.

    Returns:
        str: The base64 encoded generated image.
    """

    # Load models onto GPU if not already loaded
    load_models_gpu([final_unet, final_clip, final_vae])

    # Prepare prompts (simplified for single image)
    positive_cond = clip_encode(texts=[prompt])
    negative_cond = clip_encode(texts=[negative_prompt]) if negative_prompt else positive_cond.copy() 

    # Generate initial latent image
    initial_latent = generate_empty_latent(width=width, height=height)

    # Perform diffusion process (no refiner, simplified callback)
    image = process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=steps,
        switch=0,  # No refiner
        width=width,
        height=height,
        image_seed=seed or 0,
        callback=None,  # No callback for now
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        latent=initial_latent,
        cfg_scale=cfg_scale
    )[0]

    # Convert image to base64
    image_tensor = torch.from_numpy(image).to(intermediate_device())
    base64_image = narray_to_base64img(image_tensor)

    return base64_image

# Example usage
prompt = "A majestic lion overlooking the savanna at sunset."
image_base64 = generate_single_image(prompt)

# Display or save the image (you'll need to add code for this part)
print(image_base64) 
