from facenet_pytorch import MTCNN
import torch
import numpy as np
# import mmcv, cv2
import cv2
from PIL import Image, ImageDraw
# from IPython import display
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

# 读取图像
image_path = './inputs/IMG_20231015_104002.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
frame  = Image.fromarray(image_rgb)

# 检测人脸
boxes, _ = mtcnn.detect(frame)

# # 绘制人脸框
# frame_draw = frame.copy()
# draw = ImageDraw.Draw(frame_draw)
# for box in boxes:
#     draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

# d = display.display(frame_draw, display_id=True)

# 绘制检测结果
if boxes is not None:
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    max_area = max(areas)
    threshold_area = max_area * 0.20  # 过滤掉面积小于最大面积10%的人脸

    for box, area in zip(boxes, areas):
        if area >= threshold_area:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 5)

# 获取屏幕分辨率
screen_width = 1366  # 替换为你的屏幕宽度
screen_height = 600  # 替换为你的屏幕高度

# 获取图像尺寸
img_height, img_width, _ = image_rgb.shape

# 计算缩放比例
scale = min(screen_width / img_width, screen_height / img_height)

# 计算缩放后的尺寸
new_width = int(img_width * scale)
new_height = int(img_height * scale)

# 缩放图像
resized_image = cv2.resize(image_rgb, (new_width, new_height))

# 转换颜色格式以适应 OpenCV 和 Matplotlib
image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

# 显示图像
cv2.imshow('Face Detection', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()