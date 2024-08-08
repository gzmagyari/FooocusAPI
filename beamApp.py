import os
import ssl
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from beam import endpoint, Image, Volume, env, Output
from urllib.parse import urlparse
from typing import Optional, Dict, List

from build_launcher import build_launcher
from modules.launch_util import delete_folder_content
from apis.models.requests import CommonRequest
from modules.config import path_outputs
from apis.utils.img_utils import base64_to_image
from modules import config
from classes.FooocusModel import FooocusModel
from makeModelDictionary import makeModelDictionary
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

# Path to cache model weights
MODEL_PATH = "./models"

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
    on_start=load_model,
    image=Image(
        python_version="python3.9",
        python_packages=[
            "fastapi",
            "sqlalchemy",
            "aiofiles",
            "uvicorn",
            "Pillow",
        ],
        base_image="docker.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04",
    ),
    gpu="A100",
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
        model = load_model()
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
