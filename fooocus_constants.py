import os

VOLUME_MODEL_PATH = "/volumes/fooocus_model_cache"
VOLUME_OUTPUT_DIR = os.path.join(VOLUME_MODEL_PATH, "outputs")
VOLUME_INPUT_DIR = os.path.join(VOLUME_MODEL_PATH, "inputs")
#LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
LOCAL_MODEL_PATH = "/workspace/models"
#LOCAL_MODEL_PATH = "/root/FooocusAPI/models"
USE_VOLUME_FOR_CHECKPOINTS = False

os.makedirs(VOLUME_MODEL_PATH, exist_ok=True)
os.makedirs(VOLUME_OUTPUT_DIR, exist_ok=True)
os.makedirs(VOLUME_INPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
