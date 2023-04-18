# Reference：https://www.bilibili.com/video/BV1GR4y1W7KS/?spm_id_from=333.337.search-card.all.click&vd_source=d1982582e601460dfd542d7ff0efae87


import time

import cv2  # 用于计算机视觉任务的OpenCV库
import mediapipe as mp  # 用于构建实时应用程序的MediaPipe库

# 连接到系统上的默认相机
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.4
)
'''
Hands参数：
static_image_mode=False,                      # 如果侦测一张图片，就写True；如果是动态的，就False
               max_num_hands=2,               # 最多检测的手的数目
               model_complexity=1,            # 模型的复杂度，只能设置成0或者1，1比较精准
               min_detection_confidence=0.5,   # 最低的侦测置信度
               min_tracking_confidence=0.5     # 最低的追踪置信度
'''
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
pTime = 0
cTime = 0

if __name__ == '__main__':

    # 开始一个无限循环，连续从相机读取帧并将其显示出来
    while True:
        # 从相机读取一帧
        ret, img = cap.read()

        # 如果成功读取了帧，则使用OpenCV的cv2.imshow函数在窗口中显示它
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            # print(result.multi_hand_landmarks)
            imgHeight = img.shape[0]
            imgWidth = img.shape[1]
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    # print(handLms)
                    # landmark
                    # {
                    #     x: 0.85913503
                    #     y: 0.28609425
                    #     z: 0.05231086
                    # }
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    for i, lm in enumerate(handLms.landmark):
                        xPos = int(lm.x * imgWidth)
                        yPos = int(lm.y * imgHeight)
                        # cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        if i == 4:
                            cv2.circle(img, (xPos, yPos), 10, (166, 32, 56), cv2.FILLED)
                        print(f"{i} - ({xPos}, {yPos})")

            # 测算FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3)

            cv2.imshow('img', img)

        # 等待按键事件，并检查是否按下了“q”键。如果是，则跳出循环
        if cv2.waitKey(1) == ord('q'):
            break
