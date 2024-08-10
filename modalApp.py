import modal
import asyncio
import os
import ssl
import uuid
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, List
from urllib.parse import urlparse
import shutil

# Define Modal app
app = modal.App("fooocus-ai-service")

# Define the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "uvicorn",
        "fastapi",
        "sqlalchemy",
        "aiofiles",
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
        "pydantic"
    ).
    copy_local_dir("./apis", "/root/apis")
    .copy_local_dir("./classes", "/root/classes")
    .copy_local_dir("./css", "/root/css")
    .copy_local_dir("./docs", "/root/docs")
    .copy_local_dir("./extras", "/root/extras")
    .copy_local_dir("./javascript", "/root/javascript")
    .copy_local_dir("./language", "/root/language")
    .copy_local_dir("./ldm_patched", "/root/ldm_patched")
    .copy_local_dir("./modules", "/root/modules")
    .copy_local_dir("./presets", "/root/presets")
    .copy_local_dir("./sdxl_styles", "/root/sdxl_styles")
    .copy_local_dir("./tests", "/root/tests")
    .copy_local_dir("./utils", "/root/utils")
    .copy_local_dir("./wildcards", "/root/wildcards")
    .copy_local_dir("./models/prompt_expansion", "/root/models/prompt_expansion")
    .copy_local_file("./args_manager.py", "/root/args_manager.py")
    .copy_local_file("./build_launcher.py", "/root/build_launcher.py")
    .copy_local_file("./fooocus_constants.py", "/root/fooocus_constants.py")
    .copy_local_file("./fooocus_version.py", "/root/fooocus_version.py")
    .copy_local_file("./makeModelDictionary.py", "/root/makeModelDictionary.py")
    .copy_local_file("./shared.py", "/root/shared.py")
    .copy_local_file("./webui.py", "/root/webui.py")
    .run_commands("pip uninstall -y pydantic")
    .run_commands("pip install pydantic")
    .run_commands("pip uninstall -y fastapi")
    .run_commands("pip install fastapi")
)

# with image.imports():
#     from apis.models.requests import CommonRequest
#     from apis.utils.img_utils import base64_to_image
#     from classes.FooocusModel import FooocusModel
#     from makeModelDictionary import makeModelDictionary
#     import fooocus_constants

fastapi_app = FastAPI()

# Define the Volume correctly
model_volume = modal.Volume.from_name("fooocus_model_cache", create_if_missing=True)

# Define the class to manage the model
@app.cls(gpu="A10G", container_idle_timeout=240, image=image, volumes={"/fooocus_model_cache": model_volume})
class FooocusModelManager:
    
    @modal.enter()
    def initializeApp(self):
        from classes.FooocusModel import FooocusModel
        from makeModelDictionary import makeModelDictionary
        import fooocus_constants

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ.setdefault("GRADIO_SERVER_PORT", "7865")
        ssl._create_default_https_context = ssl._create_unverified_context
        
        file_dict = makeModelDictionary(fooocus_constants.VOLUME_MODEL_PATH, fooocus_constants.LOCAL_MODEL_PATH, fooocus_constants.USE_VOLUME_FOR_CHECKPOINTS)
        self.download_files(file_dict)
        self.copy_local_directory(os.path.join("./models", "prompt_expansion"), os.path.join(fooocus_constants.VOLUME_MODEL_PATH, "prompt_expansion"))
        self.model = FooocusModel()
        asyncio.run(self.model.startInBackground())

    def load_file_from_url(self, url: str, *, model_dir: str, progress: bool = True, file_name: Optional[str] = None) -> str:
        """Download a file from `url` into `model_dir`, using the file present if possible."""
        domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
        url = str.replace(url, "https://huggingface.co", domain, 1)
        os.makedirs(model_dir, exist_ok=True)
        if not file_name:
            parts = urlparse(url)
            file_name = os.path.basename(parts.path)
        cached_file = os.path.abspath(os.path.join(model_dir, file_name))
        if not os.path.exists(cached_file):
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
        return cached_file

    def copy_local_directory(self, source: str, destination: str):
        """Copy models from source to destination directory."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"The source directory {source} does not exist.")
        if not os.path.exists(destination):
            os.makedirs(destination)
        for item in os.listdir(source):
            src_path = os.path.join(source, item)
            dest_path = os.path.join(destination, item)
            if os.path.isdir(src_path):
                try:
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                except FileExistsError:
                    pass
            else:
                try:
                    shutil.copy2(src_path, dest_path)
                except shutil.SameFileError:
                    pass

    def download_files(self, file_dict: Dict[str, List[str]]):
        for directory, urls in file_dict.items():
            for url in urls:
                if "@@" in url:
                    url, file_name = url.split("@@")
                    self.load_file_from_url(url, model_dir=directory, file_name=file_name)
                else:
                    self.load_file_from_url(url, model_dir=directory)

    @modal.method()
    async def generate_image(self, prompt: str, negative_prompt: str = None, width: int = 512, height: int = 512, performance_selection: str = "Quality"):
        from apis.models.requests import CommonRequest
        from apis.utils.img_utils import base64_to_image
        import fooocus_constants

        request = CommonRequest(prompt=prompt, negative_prompt=negative_prompt, performance_selection=performance_selection)
        result = await self.model.async_worker(request=request, wait_for_result=True)
        if "base64_result" in result:
            base64str = result["base64_result"][0]

            return {"image": base64str}
        else:
            raise HTTPException(status_code=500, detail="Image generation failed")

# FastAPI endpoint
@fastapi_app.post("/generate_image")
async def generate_image_endpoint(request: dict):
    manager = FooocusModelManager()
    result = await manager.generate_image(request["prompt"], request["negative_prompt"], request["width"], request["height"], request["performance_selection"])
    #result = manager.generate_image.remote(prompt, negative_prompt, width, height, performance_selection)
    return result

# Modal ASGI app
@app.function()
@modal.asgi_app()
def fastapi_asgi_app():
    return fastapi_app
