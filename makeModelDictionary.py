import os

def makeModelDictionary(modelPath):
    checkPointPath = os.path.join(modelPath, "checkpoints")
    loraPath = os.path.join(modelPath, "loras")
    inpaintPath = os.path.join(modelPath, "inpaint")
    controlNetPath = os.path.join(modelPath, "controlnet")
    upscaleModelsPath = os.path.join(modelPath, "upscale_models")
    clipVisionPath = os.path.join(modelPath, "clip_vision")
    vaeApproxPath = os.path.join(modelPath, "vae_approx")
    promptExpansionPath = os.path.join(modelPath, "prompt_expansion/fooocus_expansion")
    safetyCheckerPath = os.path.join(modelPath, "safety_checker")

    # checkPointPath = "/models/checkpoints"
    # loraPath = "/models/loras"
    # inpaintPath = "/models/inpaint"
    # controlNetPath = "/models/controlnet"
    # upscaleModelsPath = "/models/upscale_models"
    # clipVisionPath = "/models/clip_vision"
    # vaeApproxPath = "/models/vae_approx"
    # promptExpansionPath = "/models/prompt_expansion/fooocus_expansion"
    # safetyCheckerPath = "/models/safety_checker"

    return {
        checkPointPath: [
            "https://huggingface.co/roguemcocdx/ModelsXL/resolve/main/pandorasBoxNSFW_v1PussBoots.safetensors?download=true"
        ],
        loraPath: [
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/sdxl_lcm_lora.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/sdxl_lightning_4step_lora.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/sdxl_hyper_sd_4step_lora.safetensors?download=true"
        ],
        inpaintPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_inpaint_head.pth?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/inpaint.fooocus.patch?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/inpaint_v25.fooocus.patch?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/inpaint_v26.fooocus.patch?download=true"
        ],
        controlNetPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/control-lora-canny-rank128.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_xl_cpds_128.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_ip_negative.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/ip-adapter-plus_sdxl_vit-h.bin?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/ip-adapter-plus-face_sdxl_vit-h.bin?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/detection_Resnet50_Final.pth?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/detection_mobilenet0.25_Final.pth?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/parsing_parsenet.pth?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/parsing_bisenet.pth?download=true"
        ],
        upscaleModelsPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_upscaler_s409985e5.bin?download=true"
        ],
        clipVisionPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/clip_vision_vit_h.safetensors?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/model_base_caption_capfilt_large.pth?download=true"
        ],
        vaeApproxPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/xlvaeapp.pth?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/vaeapp_sd15.pt?download=true",
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/xl-to-v1_interposer-v3.1.safetensors?download=true",
            "https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors"
        ],
        promptExpansionPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_expansion.bin?download=true@@pytorch_model.bin"
        ],
        safetyCheckerPath: [
            "https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/stable-diffusion-safety-checker.bin?download=true"
        ]
    }
