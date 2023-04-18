"""
Simultaneously detect body posture and hand posture.
"""

import cv2
import mediapipe as mp

# 初始化mediapipe模型对象
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

# 初始化cv2 VideoCapture对象，打开摄像头
cap = cv2.VideoCapture(0)

# 设置帧大小（可更改为所需大小）
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 初始化视频写入器，用于将检测结果保存为输出视频
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame_width, frame_height))

# 初始化姿势检测和手势检测对象并设置参数
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        # 读取一帧图像
        ret, image = cap.read()

        # 将图像翻转以显示正确方向
        image = cv2.flip(image, 1)

        # 转换图像色彩空间为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 进行姿势检测并绘制标记点
        pose_results = pose.process(image_rgb)
        if pose_results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 进行手势检测并绘制标记点
        hands_results = hands.process(image_rgb)
        if hands_results.multi_hand_landmarks is not None:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 显示处理后的图像
        cv2.imshow('MediaPipe Hands and Pose', image)

        # 将处理后的图像写入输出视频
        out.write(image)

        # 按下q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
