import cv2
from compares_faces import *

def predict_face_people(model_face_person, img, classes=[], conf=0.4):
    if classes:
        results = model_face_person.predict(img, classes=classes, conf=conf, device='cuda')
    else:
        results = model_face_person.predict(img, conf=conf, device='cuda')
    return results

def detect_face_people(model_face_person, model_face_id, class_face_id, img, classes=[], conf=0.4):
    results = predict_face_people(model_face_person, img, classes, conf)
    faces = []
    bounding_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            if label == "face":
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                face_img = img[y1:y2, x1:x2]

                faces.append(face_img)
                bounding_boxes.append([x1, y1, x2, y2])
            elif label == "person":
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if faces:
        results = compare_faces(faces, model_face_id, class_face_id)
        for (name, prob), (x1, y1, x2, y2) in zip(results, bounding_boxes):
            if prob > 0.75:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                cv2.putText(img, str(round(prob, 3)), (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)

    return img