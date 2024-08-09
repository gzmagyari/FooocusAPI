import os
import asyncio
import json
from beam import endpoint, Image, Volume, env, Output
from urllib.parse import urlparse
from typing import Optional, Dict, List
from classes.FooocusModel import FooocusModel
from apis.models.requests import CommonRequest
from apis.utils.img_utils import base64_to_image
from makeModelDictionary import makeModelDictionary

def initializeApp():
    import os
    import ssl
    from build_launcher import build_launcher
    from modules import config
    import os
    import sys

    print('[System ARGV] ' + str(sys.argv))

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root)
    os.chdir(root)

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    if "GRADIO_SERVER_PORT" not in os.environ:
        os.environ["GRADIO_SERVER_PORT"] = "7865"

    ssl._create_default_https_context = ssl._create_unverified_context

    def ini_args():
        from args_manager import args
        return args

    build_launcher()
    try:
        args = ini_args()
    except:
        pass

    if args.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
        print("Set device to:", args.gpu_device_id)

    os.environ['GRADIO_TEMP_DIR'] = config.temp_path

    return load_model()
    

# Path to cache model weights
MODEL_PATH = "/models"

def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file

def download_files(file_dict: Dict[str, List[str]]):
    for directory, urls in file_dict.items():
        for url in urls:
            load_file_from_url(url, model_dir=directory)

def load_model():
    file_dict = makeModelDictionary(MODEL_PATH)
    download_files(file_dict)
    model = FooocusModel()
    asyncio.run(model.startInBackground())
    return model

volume = Volume(name="fooocus_model_cache", mount_path=MODEL_PATH)

@endpoint(
    name="fooocus-ai-service",
    volumes=[volume],
    on_start=initializeApp,
    image=Image(
        python_version="python3.10",
        python_packages=[
            "fastapi",
            "sqlalchemy",
            "aiofiles",
            "uvicorn",
            "Pillow==9.4.0",
            "torchsde==0.2.6",
            "einops==0.4.1",
            "transformers==4.30.2",
            "safetensors==0.3.1",
            "accelerate==0.21.0",
            "pyyaml==6.0",
            "scipy==1.9.3",
            "tqdm==4.64.1",
            "psutil==5.9.5",
            "pytorch_lightning==1.9.4",
            "omegaconf==2.2.3",
            "gradio==3.41.2",
            "pygit2==1.12.2",
            "opencv-contrib-python==4.8.0.74",
            "httpx==0.24.1",
            "onnxruntime==1.16.3",
            "timm==0.9.2",
            "sse_starlette",
            "sqlalchemy",
            "httpx"
        ],
        base_image="docker.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04",
        commands=[
            # Install libGL and other dependencies
            "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0"
        ]
    ),
    gpu="A10G",
    cpu=2,
    memory="16Gi",
)
def generate_image(context, prompt: str, negative_prompt: str = None, width: int = 512, height: int = 512, performance_selection: str = "Quality"):
    model = context.on_start_value

    request = CommonRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        performance_selection=performance_selection
    )

    async def main():
        result = await model.async_worker(request=request, wait_for_result=True)
        return result

    result = asyncio.run(main())
    
    # Save result image
    if "base64_result" in result:
        base64str = result["base64_result"][0]
        image = base64_to_image(base64str, "./result.png")
        output = Output.from_pil_image(image)
        output.save()
        url = output.public_url(expires=400)
        print(url)
        return {"image": url}
    else:
        return {"error": "Image generation failed"}

if __name__ == "__main__":
    if env.is_remote():
        generate_image.serve()
    else:
        model = initializeApp()
        request = CommonRequest(
            prompt="a cute cat, crisp clear, 4k, vivid colors, high resolution",
            negative_prompt="blurry, low resolution, pixelated",
            performance_selection="Quality"
        )
        
        async def main():
            result = await model.async_worker(request=request, wait_for_result=True)
            return result

        result = asyncio.run(main())
        if "base64_result" in result:
            base64str = result["base64_result"][0]
            base64_to_image(base64str, "./result.png")
            print("Image saved as result.png")
        else:
            print("Image generation failed")
