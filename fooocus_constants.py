import os

#VOLUME_MODEL_PATH = "/volumes/fooocus_model_cache"
#VOLUME_MODEL_PATH = "/fooocus_model_cache"
VOLUME_MODEL_PATH = "/root/models"
VOLUME_OUTPUT_DIR = os.path.join(VOLUME_MODEL_PATH, "outputs")
VOLUME_INPUT_DIR = os.path.join(VOLUME_MODEL_PATH, "inputs")
#LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
#LOCAL_MODEL_PATH = "/workspace/models"
#LOCAL_MODEL_PATH = "/fooocus_model_cache"
LOCAL_MODEL_PATH = "/root/models"
#LOCAL_MODEL_PATH = "/root/FooocusAPI/models"
USE_VOLUME_FOR_CHECKPOINTS = False

os.makedirs(VOLUME_MODEL_PATH, exist_ok=True)
os.makedirs(VOLUME_OUTPUT_DIR, exist_ok=True)
os.makedirs(VOLUME_INPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

ENABLE_GPU_LOGS = False