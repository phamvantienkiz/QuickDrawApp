import numpy as np
import os
from src.config import DATA_DIR, CLASSES, IMG_SIZE
import tensorflow as tf

def load_limited_data(max_samples_per_class=10000):
    """
    Load một số lượng giới hạn dữ liệu (max_samples_per_class mỗi lớp) và tạo generator
    để tránh tải toàn bộ vào RAM.
    """
    def parse_data(file_path, max_samples):
        # Load dữ liệu từ file .npy
        data = np.load(file_path)
        # Lấy ngẫu nhiên max_samples mẫu (nếu số mẫu trong file < max_samples thì lấy hết)
        num_samples = min(max_samples, data.shape[0])
        indices = np.random.choice(data.shape[0], num_samples, replace=False)
        data = data[indices]
        # Chuẩn hóa dữ liệu (0-255 thành 0-1)
        data = data.astype('float32') / 255.0
        # Reshape thành định dạng (samples, 28, 28, 1)
        data = data.reshape((num_samples, 28, 28, 1))
        return data

    # Tạo danh sách dữ liệu và nhãn
    all_images = []
    all_labels = []

    for idx, class_name in enumerate(CLASSES):
        file_path = os.path.join(DATA_DIR, f"full_numpy_bitmap_{class_name}.npy")
        if os.path.exists(file_path):
            images = parse_data(file_path, max_samples_per_class)
            all_images.append(images)
            all_labels.append(np.full((len(images),), idx))  # Tạo nhãn cho mỗi mẫu

    # Kết hợp tất cả dữ liệu
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels = tf.keras.utils.to_categorical(all_labels, num_classes=len(CLASSES))

    # Chia dữ liệu thành tập train và test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def create_dataset(X, y, batch_size=32, shuffle=True):
    """
    Tạo tf.data.Dataset để load dữ liệu theo lô, tối ưu hóa cho huấn luyện
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Tối ưu hóa prefetch để giảm thời gian chờ
    return dataset

if __name__ == "__main__":
    # Load dữ liệu với giới hạn 1000 mẫu mỗi lớp
    X_train, X_test, y_train, y_test = load_limited_data(max_samples_per_class=1000)
    print(f"Training data shape: {X_train.shape}")  # (16000, 28, 28, 1) nếu 1000 mẫu/lớp
    print(f"Test data shape: {X_test.shape}")      # (4000, 28, 28, 1)

    # Tạo dataset cho huấn luyện
    train_dataset = create_dataset(X_train, y_train, batch_size=32)
    test_dataset = create_dataset(X_test, y_test, batch_size=32, shuffle=False)

    # In một batch để kiểm tra
    for images, labels in train_dataset.take(1):
        print(f"Batch images shape: {images.shape}")  # (32, 28, 28, 1)
        print(f"Batch labels shape: {labels.shape}")  # (32, 20)