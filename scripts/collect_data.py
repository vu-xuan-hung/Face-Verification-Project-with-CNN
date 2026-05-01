###
# 1. tìm khuôn mặt trong video
# 2. vẽ tâm ở trung tâm khuôn mặt
# 3 lấy data có blud khuôn mặt >60% vì người thật, image thì không blud
# Face Detection
#    ↓
# Resize (128x128)
#    ↓
# Check:
#    - Blur
#    - Blink# ear
#    - Movement#mog2
#    ↓
# collect data: center, width, height, blur, blink, movement
###
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time
import mediapipe as mp
import numpy as np
import os

################
BLINK_THRESHOLD = 0.2
percent = 10
confident = 0.8
camWidth = 640
camHeight = 480
floatingPoint = 6
save = True
blurThreshold = 35  # giá trị blur để phân biệt giữa người thật và ảnh
outputFolderPath = "../data/DataCollect"
classID = 0  # 0 fake, 1 real
################
cap = cv2.VideoCapture(0)
detector = FaceDetector()
cap.set(3, camWidth)
cap.set(4, camHeight)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)


# ----ham tinh EAR để phát hiện nháy mắt---
def EAR(eye):
    def dist(a, b):
        return np.linalg.norm(a - b)

    return (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * dist(eye[0], eye[3]))


outputFolderPath = "DataCollect"  # Đổi thành đường dẫn đơn giản để test
if not os.path.exists(outputFolderPath):
    os.makedirs(outputFolderPath, exist_ok=True)
while True:
    success, img = cap.read()
    imgOut = img.copy()

    if not success:
        print("Không đọc được camera")
        break

    # movement detection using MOG2
    fg_mask = backSub.apply(img)
    _, fg_mask = cv2.threshold(
        fg_mask, 250, 255, cv2.THRESH_BINARY
    )  # Loại bỏ bóng (màu xám)
    movement_value = np.sum(fg_mask) / 255
    print(f"Movement value: {movement_value}")
    img, bboxs = detector.findFaces(img, draw=False)
    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            print("Khuôn mặt được phát hiện tại: ", x, y, w, h)
            listBlur = []  # T F value if face blur or not
            listInfo = []  # NOr value->.txt
            if score > confident:
                # ---thêm phần mở rộng cho khuôn mặt để lấy nhiều data hơn---
                setw = w * percent / 100
                x = int(x - setw)
                w = int(w + 2 * setw)

                seth = h * percent / 100
                y = int(y - seth * 3)
                h = int(h + 3.5 * seth)
                # --- Clip biên ảnh (đảm bảo vùng nằm trong khung hình) ---
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                if w <= 0 or h <= 0:
                    continue
                # -----find blud-----
                imgFace = img[y : y + h, x : x + w]
                if imgFace.size == 0:
                    continue
                imgFacecopy = imgFace.copy()
                imgFace = cv2.resize(imgFacecopy, (128, 128))
                imgGray = cv2.cvtColor(imgFacecopy, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(
                    imgGray, cv2.CV_64F
                ).var()  # sharp->var cao, blur->var thấp
                if blur > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                # --find blink and movement--
                # blink detection using mediapipe face mesh
                # --- Tính blink bằng MediaPipe trên ảnh đã resize 128x128 ---
                rgb = cv2.cvtColor(imgFace, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                blink = False
                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        landmarks = face_landmarks.landmark

                        # Chỉ số landmark cho mắt trái và phải (theo MediaPipe)
                        left_eye_idx = [33, 160, 158, 133, 153, 144]
                        right_eye_idx = [362, 385, 387, 263, 373, 380]

                        def get_eye_points(idx_list):
                            pts = []
                            for i in idx_list:
                                px = int(landmarks[i].x * 128)
                                py = int(landmarks[i].y * 128)
                                pts.append(np.array([px, py]))
                            return pts

                        left_eye = get_eye_points(left_eye_idx)
                        right_eye = get_eye_points(right_eye_idx)

                        ear_left = EAR(left_eye)
                        ear_right = EAR(right_eye)
                        ear = min(
                            ear_left, ear_right
                        )  # lấy giá trị nhỏ hơn (mắt nhắm nhiều hơn)

                        if ear < BLINK_THRESHOLD:
                            blink = True
                        print(f"EAR: {ear:.3f}, Blink: {blink}")

                # ---normalize----
                imgh, imgw, _ = img.shape
                # center
                xc = x + w / 2
                yc = y + h / 2

                xcn = xc / imgw
                ycn = yc / imgh
                wn = w / imgw
                hn = h / imgh

                xcn = np.clip(xcn, 0, 1)
                ycn = np.clip(ycn, 0, 1)
                wn = np.clip(wn, 0, 1)
                hn = np.clip(hn, 0, 1)
                if x + w > imgw:
                    w = imgw - x
                if y + h > imgh:
                    h = imgh - y
                print(f"center: {xcn}, {ycn}, width: {wn}, height: {hn}, blur: {blur}")
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ---vẽ hình chữ nhật quanh khuôn mặt---
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

                cvzone.putTextRect(
                    img,
                    f"Score: {int(score * 100)}% Blur: {blur}",
                    (x, y),
                    scale=2,
                    thickness=1,
                    offset=10,
                )

                # ---collect data---
                if save:
                    if all(listBlur) and listBlur != []:
                        face_clean = imgOut[y : y + h, x : x + w]
                        timeNow = time()
                        timeNow = str(timeNow).split(".")
                        timeNow = timeNow[0] + timeNow[1]
                        # ------  Save Image  --------
                        if face_clean.size > 0:
                            cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", face_clean)

                        # ------  Save Label Text File  --------
                        for info in listInfo:
                            with open(f"{outputFolderPath}/{timeNow}.txt", "a") as f:
                                f.write(info)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
