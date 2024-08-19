from detect_face_peple import *

def multi_task_model(model_face_person, model_face_id, class_face_id, img, conf, classes=[]):
    results = predict_face_people(model_face_person, img, conf, classes)
    faces = []
    bounding_boxe_faces = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            if label == "face":
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                face_img = img[y1:y2, x1:x2]
                faces.append(face_img)
                bounding_boxe_faces.append([x1, y1, x2, y2])

            elif label == "person":
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    if faces:
        results = compare_faces(faces, model_face_id, class_face_id)
        for (name, prob), (x1, y1, x2, y2) in zip(results, bounding_boxe_faces):
            if prob > config.IOU_FACE_ID:
                if y2 > config.LINE_POSITION:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                    cv2.putText(img, str(round(prob, 3)), (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                elif y2 < config.LINE_POSITION:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cv2.putText(img, "", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
            else:
                if y2 > config.LINE_POSITION:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cv2.putText(img, "Doan Xem!", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                elif y2 < config.LINE_POSITION:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cv2.putText(img, "", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)

    return img