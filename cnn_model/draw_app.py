import cv2
import numpy as np
import tensorflow as tf
from src.config import CLASSES, TRAINED_MODELS_DIR
import os

# Load mô hình đã huấn luyện
model_path = os.path.join(TRAINED_MODELS_DIR, "quickdraw_cnn_optimized_best.h5")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


# Hàm dự đoán
def predict_drawing(drawing):
    # Chuyển đổi hình vẽ thành mảng numpy (28x28, grayscale)
    drawing = cv2.resize(drawing, (28, 28), interpolation=cv2.INTER_AREA)
    drawing_array = np.array(drawing, dtype=np.float32)
    # Áp dụng ngưỡng để làm rõ hình
    drawing_array = np.where(drawing_array > 128, 255, 0).astype(np.uint8)
    drawing_array = drawing_array / 255.0  # Chuẩn hóa 0-1
    drawing_array = drawing_array.reshape(1, 28, 28, 1)  # Định dạng cho mô hình
    prediction = model.predict(drawing_array, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    print("Xác suất từng lớp:", dict(zip(CLASSES, prediction[0])))  # Debug
    return predicted_class, confidence


# Hàm xử lý sự kiện chuột để vẽ
def paint_draw(event, x, y, flags, param):
    global ix, iy, is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            cv2.line(canvas, (ix, iy), (x, y), (255, 255, 255), 5)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        cv2.line(canvas, (ix, iy), (x, y), (255, 255, 255), 5)


# Khởi tạo canvas và biến toàn cục
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
ix, iy = -1, -1
is_drawing = False

# Tạo cửa sổ OpenCV và bind sự kiện chuột
cv2.namedWindow("QuickDraw Canvas")
cv2.setMouseCallback("QuickDraw Canvas", paint_draw)

# Vòng lặp chính
while True:
    # Hiển thị canvas (đảo màu để nét vẽ trắng thành đen và nền đen thành trắng)
    display_image = 255 - canvas.copy()

    # Hiển thị hướng dẫn
    cv2.putText(display_image, "Space: Predict | C: Clear | Esc: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("QuickDraw Canvas", display_image)
    key = cv2.waitKey(1) & 0xFF

    # Phím Space: Dự đoán
    if key == ord(" "):
        # Chuyển canvas sang grayscale
        canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Tìm vùng có nét vẽ để cắt
        ys, xs = np.nonzero(canvas_gs)
        if len(ys) > 0 and len(xs) > 0:
            min_y, max_y = np.min(ys), np.max(ys)
            min_x, max_x = np.min(xs), np.max(xs)

            # Cắt và resize vùng có nét vẽ
            cropped_image = canvas_gs[min_y:max_y, min_x:max_x]
            if cropped_image.size > 0:
                # Thêm padding để đảm bảo hình vuông
                height, width = cropped_image.shape
                max_dim = max(height, width)
                padded_image = np.zeros((max_dim, max_dim), dtype=np.uint8)
                y_offset = (max_dim - height) // 2
                x_offset = (max_dim - width) // 2
                padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image

                # Dự đoán
                predicted_class, confidence = predict_drawing(padded_image)

                # Hiển thị kết quả trên canvas
                result_text = f"Predicted: {predicted_class} ({confidence:.2f})"
                cv2.putText(display_image, result_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("QuickDraw Canvas", display_image)
                cv2.waitKey(2000)  # Hiển thị kết quả trong 2 giây

                # Reset canvas sau khi dự đoán
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                ix, iy = -1, -1
        else:
            print("Không có nét vẽ để dự đoán!")

    # Phím C: Xóa canvas
    elif key == ord("c"):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        ix, iy = -1, -1

    # Phím Esc: Thoát
    elif key == 27:
        break

cv2.destroyAllWindows()