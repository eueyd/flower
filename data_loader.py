# data_loader.py
"""
数据加载模块
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR


def get_data_transforms():
    """获取数据增强和变换"""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def create_data_loaders(batch_size=BATCH_SIZE, num_workers=0):
    """创建数据加载器"""
    train_transform, test_transform = get_data_transforms()

    # 检查数据目录
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"训练目录不存在: {TRAIN_DIR}")
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"测试目录不存在: {TEST_DIR}")

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader, train_dataset, test_dataset


def get_dataset_info():
    """获取数据集信息"""
    train_transform, test_transform = get_data_transforms()

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

    info = {
        'num_classes': len(train_dataset.classes),
        'class_names': train_dataset.classes,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': {v: k for k, v in train_dataset.class_to_idx.items()}
    }

    return info


def verify_data_loading():
    """验证数据加载"""
    try:
        train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(batch_size=4)

        print("✅ 数据加载验证成功!")
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        print(f"类别数: {len(train_dataset.classes)}")

        # 检查一个batch
        images, labels = next(iter(train_loader))
        print(f"\nBatch 信息:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签值示例: {labels[:5].tolist()}")
        print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False