import os

DATA_DIR = os.path.join("dataset","Images") # sökväg av datakatalog
IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 16
EPOCHS = 10
CLASSES = 2
TRAINED_MODEL_DIR = os.path.join("VGGmodel","models")
CHECKPOINT_DIR = os.path.join("VGGmodel","checkpoints")
AUGMENTATION = False
BASE_LOG_DIR = "base_log_dir"




