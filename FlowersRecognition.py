"""
=========================== 环境配置要求 ===========================
1. Python版本：3.12
2. 依赖：
   - PyTorch：2.9.1+cu128（需匹配CUDA版本）
   - torchvision（与PyTorch版本对应）
   - matplotlib（可视化图表）
   - numpy
   - tqdm（进度条显示）
   - pillow（图像加载）
3. CUDA配置（GPU加速）：
   - CUDA Toolkit：12.8
=========================== 代码功能说明 ===========================
- 模式1（重新训练）：完整训练流程，保存模型权重+训练指标，自动可视化
- 模式2（仅测试）：加载已有模型，快速测试并展示结果+可视化
- 数据路径：需保证项目根目录下有"flowers"文件夹，包含train/test子文件夹
- 输出文件：
  - 模型权重：best_flower_model.pth
  - 训练指标：training_metrics.npy
  - 可视化图表：loss_curve.png/accuracy_curve.png/class_distribution.png/final_accuracy.png/confusion_matrix.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
# 模型/数据保存路径
MODEL_PATH = "best_flower_model.pth"
METRICS_PATH = "training_metrics.npy"
# 类别映射
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
NUM_CLASSES = len(CLASS_NAMES)
# 训练参数（全局共享）
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)


# ===================== 工具函数 =====================
def check_gpu_availability():
    """验证GPU是否可用并输出详细信息"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("=" * 50)
        print("GPU加速已启用！")
        print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch CUDA是否可用: {torch.cuda.is_available()}")
        print("=" * 50)
    else:
        device = torch.device("cpu")
        print("=" * 50)
        print("警告：未检测到GPU，使用CPU训练/测试（速度较慢）！")
        print("=" * 50)
    return device


def load_data():
    """加载并预处理数据集（训练/测试共享）"""
    BASE_DIR = "flowers"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    TEST_DIR = os.path.join(BASE_DIR, "test")

    # 训练集变换（含数据增强）
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 测试集变换（无数据增强）
    test_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 打印数据集信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别数: {NUM_CLASSES}, 类别名称: {CLASS_NAMES}")

    return train_loader, test_loader, train_dataset, test_dataset


def build_model(device):
    """构建迁移学习模型（训练/测试共享）"""
    # 加载预训练的ResNet50
    model = models.resnet50(pretrained=True)
    # 冻结特征提取层
    for param in model.parameters():
        param.requires_grad = False
    # 替换全连接层
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES)
    )
    # 移至指定设备
    model = model.to(device)
    # 验证模型设备
    print(f"模型已加载至: {next(model.parameters()).device}")
    return model


def save_training_metrics(train_losses, train_accs, test_losses, test_accs):
    """保存训练指标到文件"""
    metrics = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }
    np.save(METRICS_PATH, metrics)
    print(f"训练指标已保存至: {METRICS_PATH}")


def load_training_metrics():
    """加载训练指标文件"""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"未找到训练指标文件: {METRICS_PATH}，请先执行训练模式！")
    metrics = np.load(METRICS_PATH, allow_pickle=True).item()
    return metrics["train_losses"], metrics["train_accs"], metrics["test_losses"], metrics["test_accs"]


def visualize_results(train_losses, train_accs, test_losses, test_accs, train_dataset, test_dataset):
    """可视化分析（独立图表+单独保存）"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 计算类别分布
    TRAIN_DIR = os.path.join("flowers", "train")
    TEST_DIR = os.path.join("flowers", "test")
    train_class_counts = [len(os.listdir(os.path.join(TRAIN_DIR, cls))) for cls in CLASS_NAMES]
    test_class_counts = [len(os.listdir(os.path.join(TEST_DIR, cls))) for cls in CLASS_NAMES]
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]

    # 图表1：损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="训练损失", marker="o")
    plt.plot(range(1, EPOCHS + 1), test_losses, label="测试损失", marker="s")
    plt.title("训练与测试损失变化", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("损失值", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

    # 图表2：准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_accs, label="训练准确率", marker="o", color="green")
    plt.plot(range(1, EPOCHS + 1), test_accs, label="测试准确率", marker="s", color="red")
    plt.title("训练与测试准确率变化", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("准确率", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.show()

    # 图表3：类别分布
    plt.figure(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    plt.bar(x - width / 2, train_class_counts, width, label="训练集", color="skyblue")
    plt.bar(x + width / 2, test_class_counts, width, label="测试集", color="orange")
    plt.title("数据集类别分布", fontsize=14)
    plt.xlabel("花卉类别", fontsize=12)
    plt.ylabel("样本数量", fontsize=12)
    plt.xticks(x, CLASS_NAMES, rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300)
    plt.show()

    # 图表4：最终准确率
    plt.figure(figsize=(8, 6))
    plt.bar(["训练集", "测试集"], [final_train_acc, final_test_acc], color=["green", "red"], width=0.5)
    plt.title("最终准确率对比", fontsize=14)
    plt.ylabel("准确率", fontsize=12)
    plt.ylim(0, 1.0)
    plt.text(0, final_train_acc + 0.02, f"{final_train_acc:.4f}", ha="center")
    plt.text(1, final_test_acc + 0.02, f"{final_test_acc:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig("final_accuracy.png", dpi=300)
    plt.show()


# 绘制混淆矩阵的函数
def plot_confusion_matrix(model, test_loader, device):
    """绘制混淆矩阵图"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 初始化混淆矩阵
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    # 收集预测结果
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # 更新混淆矩阵
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf_matrix[t, p] += 1

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    # 归一化混淆矩阵（按行）
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    # 热力图展示
    plt.imshow(conf_matrix_norm, cmap='Blues')
    plt.title('混淆矩阵', fontsize=16)
    plt.colorbar(label='归一化值')
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)

    # 设置刻度标签
    plt.xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=15)
    plt.yticks(np.arange(NUM_CLASSES), CLASS_NAMES)

    # 添加数值标注
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, f'{conf_matrix_norm[i, j]:.2f}',
                     ha='center', va='center', color='black' if conf_matrix_norm[i, j] < 0.5 else 'white')

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


# ===================== 训练/测试核心逻辑 =====================
def train_model(device):
    """重新训练模型并保存结果"""
    # 加载数据
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    # 构建模型
    model = build_model(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练指标记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0.0

    # 开始训练
    print("\n========== 开始训练 ==========")
    for epoch in range(EPOCHS):
        # 训练一个epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # 验证
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Validation")
            for images, labels in pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})
        test_loss = running_loss / len(test_loader.dataset)
        test_acc = correct / total

        # 学习率衰减
        scheduler.step()

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存最优模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"保存最优模型，测试准确率: {best_test_acc:.4f}")

        # 打印epoch结果
        print(f"Epoch {epoch + 1} | 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"          | 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}\n")

    # 保存训练指标
    save_training_metrics(train_losses, train_accs, test_losses, test_accs)
    print(f"训练完成！最优测试准确率: {best_test_acc:.4f}")

    # 可视化结果
    visualize_results(train_losses, train_accs, test_losses, test_accs, train_dataset, test_dataset)

    # 新增：训练完成后绘制混淆矩阵
    plot_confusion_matrix(model, test_loader, device)

    # 最终评估
    evaluate_model(device, test_loader)


def evaluate_model(device, test_loader=None):
    """仅加载模型测试并输出结果"""
    # 加载数据（若未传入test_loader）
    if test_loader is None:
        _, test_loader, _, _ = load_data()

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}，请先执行训练模式！")

    # 构建并加载模型
    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 测试集评估
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    print("\n========== 开始测试 ==========")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 整体指标
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 分类别指标
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

            pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})

    # 计算最终指标
    final_loss = running_loss / len(test_loader.dataset)
    final_acc = correct / total

    # 打印结果
    print(f"\n测试完成！")
    print(f"整体测试损失: {final_loss:.4f}, 整体测试准确率: {final_acc:.4f}")
    print("\n各类别准确率:")
    for i in range(NUM_CLASSES):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{CLASS_NAMES[i]}: {acc:.4f} (正确数: {class_correct[i]}, 总数: {class_total[i]})")

    # 加载训练指标并可视化
    try:
        train_losses, train_accs, test_losses, test_accs = load_training_metrics()
        _, _, train_dataset, test_dataset = load_data()
        visualize_results(train_losses, train_accs, test_losses, test_accs, train_dataset, test_dataset)

        # 新增：测试完成后绘制混淆矩阵
        plot_confusion_matrix(model, test_loader, device)
    except FileNotFoundError as e:
        print(f"\n警告：{e}，仅展示测试结果，跳过可视化！")


# ===================== 主程序入口 =====================
if __name__ == '__main__':
    # 设备初始化
    device = check_gpu_availability()

    # 模式选择
    print("\n========== 花卉分类模型 ==========")
    print("请选择运行模式：")
    print("1. 重新训练模型（会覆盖原有模型和训练指标）")
    print("2. 仅测试（使用已训练的模型）")
    while True:
        try:
            choice = int(input("输入选择（1/2）："))
            if choice in [1, 2]:
                break
            else:
                print("请输入1或2！")
        except ValueError:
            print("请输入有效的数字（1/2）！")

    # 执行对应模式
    if choice == 1:
        train_model(device)
    elif choice == 2:
        evaluate_model(device)