from torch.cuda import is_available

DEVICE = 'cuda' if is_available() else 'cpu'

LEARNING_RATE = 1e-4
NUM_EPOCHS = 0
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGE_HEIGHT = 360  # 1280 originally
IMAGE_WIDTH = 540  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True

MODEL_SAVE_PATH = './models/unet/unet.pth.tar'
MODEL_RESTORE_PATH = MODEL_SAVE_PATH

TRAIN_IMG_DIR = "./data/train_images/"
TRAIN_MASK_DIR = "./data/train_masks/"
VAL_IMG_DIR = "./data/val_images/"
VAL_MASK_DIR = "./data/val_masks/"

PRED_EX_SAVE_PATH = './saved_images'
INFERENCE_IMAGE_PATH = './infer/images'
INFERENCE_SEG_PATH = './infer/segmaps'
