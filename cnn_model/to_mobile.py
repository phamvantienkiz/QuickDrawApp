import tensorflow as tf

# Đường dẫn đến mô hình đã huấn luyện
model_path = "trained_models/quickdraw_cnn_optimized_best.h5"

# Load mô hình
model = tf.keras.models.load_model(model_path)

# Chuyển đổi sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Lưu file TFLite
with open("trained_models/quickdraw_cnn_optimized_best.tflite", "wb") as f:
    f.write(tflite_model)

print("Chuyển đổi thành công sang quickdraw_cnn_optimized_best.tflite!")

# Giam dung luong
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Áp dụng quantization
# tflite_model = converter.convert()
#
# with open("trained_models/quickdraw_cnn_optimized_best_quantized.tflite", "wb") as f:
#     f.write(tflite_model)