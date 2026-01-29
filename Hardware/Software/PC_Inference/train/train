import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import RoadDataset
from model import SimpleCNN  # 确保 model.py 还是原来的 SimpleCNN
import os

# 参数配置
dataset_jungle = "G:/autonomous_car/autonomous_car/dataset/self_driving_car_dataset_jungle"
dataset_make = "G:/autonomous_car/autonomous_car/dataset/self_driving_car_dataset_make"

batch_size = 32
extra_epochs = 30  # 在 0.024 基础上再跑 30 轮
learning_rate = 0.00005  # 使用更小的初始学习率进行精修
model_save_path = "model.pth"

# 数据加载
ds1 = RoadDataset(dataset_jungle)
ds2 = RoadDataset(dataset_make)
full_dataset = ConcatDataset([ds1, ds2])
loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

print(f"数据就绪：总样本数 {len(full_dataset)}")

# 模型配置 & 断点加载

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

if os.path.exists(model_save_path):
    print(f"发现原有模型，加载 对原有模型权重进行精修训练...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    print("未发现模型文件，将从零开始训练！")

# 使用 SmoothL1Loss (也叫 Huber Loss)
# 之前的 0.024 是用 MSE 练出来的，现在换成 SmoothL1 能在小误差区间更精准
criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 训练循环
print(f"开始精修训练...")

for epoch in range(extra_epochs):
    model.train()
    total_loss = 0

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device).float()

        # 数据增强：添加极微量的噪声，防止过拟合
        noise = torch.randn_like(imgs) * 0.005
        imgs = imgs + noise

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs.squeeze(), labels)
        loss.backward()

        # 梯度裁剪：保持训练稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    scheduler.step(avg_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{extra_epochs} | Avg Loss: {avg_loss:.6f} | LR: {current_lr}")

    # 每一轮都覆盖保存最新的最优权重
    torch.save(model.state_dict(), model_save_path)

print(" 原模型精修完成！模型已更新。")
