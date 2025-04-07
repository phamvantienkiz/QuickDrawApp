# utils.py
import numpy as np
from config import CLASSES
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def evaluate_model(model_path, X_test, y_test):
    """
    Đánh giá mô hình và hiển thị kết quả
    """
    model = load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    return accuracy

def plot_training_history(history):
    """
    Vẽ biểu đồ accuracy và loss trong quá trình huấn luyện
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from src.dataset import load_limited_data
    from src.config import TRAINED_MODELS_DIR
    X_test, _, y_test, _ = load_limited_data()
    model_path = os.path.join(TRAINED_MODELS_DIR, f"quickdraw_cnn_optimized_best.h5")
    accuracy = evaluate_model(model_path, X_test, y_test)