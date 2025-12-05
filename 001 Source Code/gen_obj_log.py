from ultralytics import YOLO
import cv2
import json
import logging
from datetime import datetime, timedelta, timezone
import os

KST = timezone(timedelta(hours=9))
logger = logging.getLogger('detection_logger')
logger.setLevel(logging.INFO)

model = YOLO("../best.pt")
DETECTED_DATA = "../data/gold/json/detection_log.json"
IMAGE_DIR = "../data/gold/images"
BOXED_DIR = "../data/gold/boxed_images"

CLASS_COLORS = {
    "car": (0, 255, 0),   
    "smoke": (255, 0, 0), 
    "fire": (0, 0, 255)
}


def detect_obj():
    os.makedirs(BOXED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DETECTED_DATA), exist_ok=True)

    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
    all_results = []

    for image in image_files:
        abs_path = os.path.abspath(image)
        frame = cv2.imread(abs_path)
        if frame is None:
            print(f"이미지 읽기 실패: {abs_path}")
            continue

        current_time = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
        results = model(frame, stream=False, conf=0.7)
        r = results[0]

        all_detection_data = []
        if r.boxes:
            for box in r.boxes:
                class_name = model.names[int(box.cls[0])]
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detected_data = {
                    'timestamp': current_time,
                    'object_class': class_name,
                    'bbox': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center_x': detection_center[0],
                        'center_y': detection_center[1],
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                }
                all_detection_data.append(detected_data)
        else:
            all_detection_data.append({'No Detection': True})

        boxed_path = os.path.join(BOXED_DIR, os.path.basename(image))
        cv2.imwrite(boxed_path, frame)

        data = {'image': image, 'log': all_detection_data}
        all_results.append(data)

        with open(DETECTED_DATA, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    detect_obj()