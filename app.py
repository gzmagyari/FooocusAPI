import os
import ssl
import sys
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from build_launcher import build_launcher
from modules.launch_util import delete_folder_content
from apis.models.requests import CommonRequest
from modules.config import path_outputs
from apis.utils.img_utils import (
    base64_to_image
)
from modules import config
from classes.FooocusModel import FooocusModel

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

model = FooocusModel()

request = CommonRequest(
    prompt="a cute cat, crisp clear, 4k, vivid colors, high resolution",
    negative_prompt="blurry, low resolution, pixelated",
    performance_selection = "Quality"
)

async def main():
    await model.startInBackground()
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
