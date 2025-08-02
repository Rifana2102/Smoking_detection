from ultralytics import YOLO
import cv2

# Load model (pastikan ini model deteksi rokok)
model = YOLO("models/detection_module.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek di frame
    results = model(frame)

    # Tampilkan hasil
    annotated_frame = results[0].plot()
    cv2.imshow("Deteksi Rokok", annotated_frame)

    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
