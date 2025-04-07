# config.py
import os

# Đường dẫn cơ bản
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Danh sách các lớp (classes) từ dataset QuickDraw
CLASSES = [
    "apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope",
    "eyeglasses", "guitar", "hammer", "hat", "ice cream", "leaf", "scissors",
    "star", "t-shirt", "pants", "lightning", "tree"
]

# Các tham số huấn luyện
IMG_SIZE = (28, 28)  # Kích thước ảnh đầu vào (QuickDraw thường là 28x28)
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Đường dẫn đến các thư mục
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "..", "trained_models")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "..", "tensorboard")