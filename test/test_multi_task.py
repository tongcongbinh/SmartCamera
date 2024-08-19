import cv2
import pickle
from ultralytics import YOLO
from collections import deque
import time

from configs import config
from modules.compares_faces import *
from modules.detect_face_peple import *
from modules.multi_task_model import *
from modules.hand_rec import *

# Đọc weight mô hình detect_face_person
model_face_person = YOLO(config.YOLO_FACE_PERSON_MODEL)
model_face_person.model.to('cuda')

# Đọc weight mô hình face_id
with open(config.FACEID_AIRC_MODEL, 'rb') as file:
    model_face_id, class_face_id = pickle.load(file)

# Đọc weight mô hình nhận diện hành vi tay
min_detection_confidence = 0.7
min_tracking_confidence = 0.5
hands, keypoint_classifier = load_models(True, min_detection_confidence, min_tracking_confidence)
keypoint_classifier_labels = read_labels(config.PATH_KEYPOINT_CLASSIFIER_LABELS)
history_length = 16
point_history = deque(maxlen=history_length)

# URL của camera IP hoặc sử dụng camera local
cap = cv2.VideoCapture(config.URL_CAM)

# Tiến hành chạy mô hình
while True:
    success, img = cap.read()
    if not success:
        print("Không thể đọc từ camera.")
        break

    img = cv2.resize(img, (config.SIZE_VIDEO_CAPTURE, config.SIZE_VIDEO_CAPTURE))

    # Phát hiện và vẽ hộp bao quanh khuôn mặt và người
    begin = time.time()
    result_img = multi_task_model(model_face_person, model_face_id, class_face_id, img, conf=config.IOU_FACE_PERSON, classes=[])
    end = time.time()
    print("Độ trễ detect face and people và faceID = ", (end - begin)*1000 , "ms")

    begin_tay = time.time()
    debug_image = process_frame(img, hands, keypoint_classifier, point_history, keypoint_classifier_labels, True)
    end_tay = time.time()
    print("Độ trễ tay = ", (end_tay - begin_tay)*1000, "ms")

    # Vẽ đường giới hạn trên khung hình
    cv2.line(debug_image, (0, config.LINE_POSITION), (img.shape[1], config.LINE_POSITION), (0, 255, 255), 1)

    # Hiển thị khung hình với các hộp bao quanh và tên lớp
    cv2.imshow("Image", debug_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()