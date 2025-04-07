import tensorflow as tf
from src.config import IMG_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAINED_MODELS_DIR, TENSORBOARD_DIR
from src.dataset import load_limited_data, create_dataset
import os

def build_model():
    model = tf.keras.Sequential([
        # conv1: Conv2D(1, 32, 5) + MaxPool2d(2, 2)
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # conv2: Conv2D(32, 64, 5) + MaxPool2d(2, 2)
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Flatten
        tf.keras.layers.Flatten(),
        # Dense(3136, 256) + Dropout(0.5)
        tf.keras.layers.Dense(512, activation='relu'), #256 old
        tf.keras.layers.Dropout(0.5),
        # new
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Dense(256, 64) + Dropout(0.5)
        # tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # Dense(64, num_classes)
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    # Load dữ liệu với 1000 mẫu/lớp
    x_train, x_test, y_train, y_test = load_limited_data(max_samples_per_class=10000)
    train_dataset = create_dataset(x_train, y_train, batch_size=32)  # Batch size 16 để fits 4GB VRAM
    test_dataset = create_dataset(x_test, y_test, batch_size=32, shuffle=False)

    # Xây dựng và huấn luyện mô hình
    model = build_model()
    model.summary()

    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=1)

    # Early Stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Theo dõi validation loss
        patience=5,  # Dừng sau 3 epoch không cải thiện
        restore_best_weights=True  # Khôi phục trọng số tốt nhất
    )

    # ModelCheckpoint callback để lưu mô hình tốt nhất
    checkpoint_path = os.path.join(TRAINED_MODELS_DIR, "quickdraw_cnn_optimized_best.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',  # Theo dõi validation accuracy
        save_best_only=True,  # Chỉ lưu khi mô hình cải thiện
        mode='max',  # Tối đa hóa val_accuracy
        verbose=1  # Hiển thị thông báo khi lưu
    )

    # Huấn luyện mô hình
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback],  # Thêm checkpoint
        verbose=1
    )

    # Lưu mô hình cuối cùng
    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    final_model_path = os.path.join(TRAINED_MODELS_DIR, "quickdraw_cnn_optimized_final.h5")
    model.save(final_model_path)

    return history

if __name__ == "__main__":
    history = train_model()
    print("Training completed!")

# Kiến trúc chi tiết
# conv1: Conv2D(32, 5x5) + MaxPooling2D(2x2)
# Kích thước ảnh sau pooling: 14x14 (từ 28x28).
# conv2: Conv2D(64, 5x5) + MaxPooling2D(2x2)
# Kích thước ảnh sau pooling: 7x7.
# Tổng số đặc trưng: 64 * 7 * 7 = 3136
# Fully Connected:
# Dense(256) + Dropout(0.5)
# Dense(64) + Dropout(0.5)
# Dense(20): Đầu ra 20 lớp.
# Ước tính số tham số
# conv1: (5 * 5 * 1 * 32) + 0 (no bias) ≈ 800.
# conv2: (5 * 5 * 32 * 64) + 0 ≈ 51,200.
# fc1: (3136 * 256) + 256 ≈ 803,072.
# fc2: (256 * 64) + 64 ≈ 16,448.
# fc3: (64 * 20) + 20 ≈ 1,300.
# Tổng cộng: ~872,820 tham số
# Yêu cầu bộ nhớ
# Với batch size 16:
# Kích thước activation: ~16 * (3136 + 256 + 64 + 20) * 4 bytes ≈ 220KB.
# Gradient và weight: ~872K * 4 bytes * 2 (forward + backward) ≈ 7MB.
# Tổng VRAM: ~10-12MB mỗi batch (dư sức với 4GB VRAM).
# RAM: Với generator, chỉ load ~20MB dữ liệu tổng cộng, rất an toàn với 16GB.