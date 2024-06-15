import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import KMeans
import numpy as np
from torchvision import transforms
import shutil

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 MTCNN 和 InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 读取图像文件
image_dir = './inputs'  # 替换为你的图像目录路径
output_dir = './outputs'  # 替换为输出目录路径
os.makedirs(output_dir, exist_ok=True)

# 遍历图像文件并检测人脸
file_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
face_images = []
face_embeddings = []
image_names = []
face_boxes = []

for file_path in file_paths:
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测人脸
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is not None:
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        max_area = max(areas)
        threshold_area = max_area * 0.1  # 过滤掉面积小于最大面积10%的人脸
        
        for box, area in zip(boxes, areas):
            if area >= threshold_area:
                x1, y1, x2, y2 = map(int, box)
                face = image_rgb[y1:y2, x1:x2]
                
                # 提取特征
                face_tensor = transforms.ToTensor()(cv2.resize(face, (160, 160))).unsqueeze(0).to(device)
                embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                
                face_images.append(face)
                face_embeddings.append(embedding)
                image_names.append(file_path)
                face_boxes.append((x1, y1, x2, y2))

# 聚类
k = 40  # 设定聚类数量，根据你的需求调整
kmeans = KMeans(n_clusters=k).fit(face_embeddings)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# 设置过滤阈值（基于聚类中心的距离）
threshold = 1.0  # 设定一个阈值，根据你的需求调整

# 过滤聚类结果
filtered_faces = []
filtered_labels = []
filtered_image_names = []
filtered_boxes = []

for idx, (embedding, label, face, image_name, box) in enumerate(zip(face_embeddings, labels, face_images, image_names, face_boxes)):
    center = cluster_centers[label]
    distance = np.linalg.norm(embedding - center)
    if distance <= threshold:
        filtered_faces.append(face)
        filtered_labels.append(label)
        filtered_image_names.append(image_name)
        filtered_boxes.append(box)

# 保存聚类结果
for idx, (face, label, image_name, box) in enumerate(zip(filtered_faces, filtered_labels, filtered_image_names, filtered_boxes)):
    cluster_dir = os.path.join(output_dir, f'cluster_{label}')
    os.makedirs(cluster_dir, exist_ok=True)
    face_image_path = os.path.join(cluster_dir, f'{os.path.splitext(os.path.basename(image_name))[0]}_{idx}.jpg')
    
    x1, y1, x2, y2 = box
    h, w, _ = cv2.imread(image_name).shape

    # 扩大10%
    expand_x = int((x2 - x1) * 0.4)
    expand_y = int((y2 - y1) * 0.6)
    x1 = max(x1 - expand_x, 0)
    y1 = max(y1 - expand_y, 0)
    x2 = min(x2 + expand_x, w)
    y2 = min(y2 + expand_y, h)

    # 保存裁剪并扩展后的脸部图像
    image = cv2.imread(image_name)
    face_expanded = image[y1:y2, x1:x2]
    cv2.imwrite(face_image_path, face_expanded)

print(f'Clustering complete. Results saved to {output_dir}')
