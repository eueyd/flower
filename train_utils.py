import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gc

from config import *
from model_components import ImprovedDualPathModel


def train_model(device, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    """训练模型"""
    print(f"\n🚀 开始训练双路径模型...")

    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()

    # 加载数据
    from data_loader import create_data_loaders
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders()

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"批次大小: {BATCH_SIZE}")

    # 构建模型
    print(f"\n🔧 构建模型...")
    model = ImprovedDualPathModel(
        num_classes=NUM_CLASSES,
        num_regions=NUM_REGIONS
    )
    model = model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")

    # 优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=1e-5
    )
    # warmup_epochs = 3  # 预热1个epoch（约63个batch）
    # warmup_factor = 10.0  # 预热期学习率放大10倍（1e-3 → 1e-2）

    # def warmup_lr_scheduler(epoch, batch_idx, total_batches):
    #     """动态调整学习率"""
    #     total_warmup_steps = warmup_epochs * total_batches  # 总共的预热步数
    #     current_step = epoch * total_batches + batch_idx
    #
    #     if current_step < total_warmup_steps:
    #         # 线性预热：从0到1e-2
    #         alpha = current_step / total_warmup_steps
    #         warmup_lr = LEARNING_RATE * warmup_factor * alpha
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = warmup_lr
    #     else:
            # 预热结束，使用正常学习率
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = LEARNING_RATE



    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练指标
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0.0

    print(f"\n📈 开始训练，共 {EPOCHS} 个epoch...")

    for epoch in range(EPOCHS):
        # ===== 训练阶段 =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        fusion_stats = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            # 定期清理缓存
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss, loss_details = model.compute_loss(outputs, labels, criterion)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=0.5
            )

            optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs['final_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集融合权重
            if 'fusion_weights' in outputs:
                fusion_stats.append(outputs['fusion_weights'].mean(dim=0).cpu().detach().numpy())

            # 更新进度条
            pbar.set_postfix({
                "loss": running_loss / max(total, 1),
                "acc": correct / max(total, 1),
                **{k: v for k, v in loss_details.items() if 'loss' in k}
            })

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ===== 验证阶段 =====
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs['final_logits'], labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs['final_logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    "loss": running_loss / max(total, 1),
                    "acc": correct / max(total, 1)
                })

        test_loss = running_loss / len(test_loader.dataset)
        test_acc = correct / total

        # 学习率调整
        scheduler.step()

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 打印结果
        print(f"\n📊 Epoch {epoch + 1} 结果:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最优模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_acc': best_test_acc,
                'config': {
                    'model_type': 'improved_dual_path',
                    'num_regions': NUM_REGIONS,
                    'image_size': IMAGE_SIZE,
                    'batch_size': BATCH_SIZE,
                    'num_classes': NUM_CLASSES
                }
            }, model_path)
            print(f"  💾 保存最优模型，准确率: {best_test_acc:.4f}")

    # 保存训练指标
    metrics = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'epochs': EPOCHS
    }
    np.save(metrics_path, metrics)

    print(f"\n✅ 训练完成!")
    print(f"最佳测试准确率: {best_test_acc:.4f}")

    return model, train_losses, train_accs, test_losses, test_accs


def test_model(device, model_path=MODEL_PATH):
    """测试模型"""
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练模型！")
        return

    print(f"\n🧪 测试模型...")

    # 加载数据
    from data_loader import create_data_loaders
    _, test_loader, _, test_dataset = create_data_loaders()

    # 构建模型
    model = ImprovedDualPathModel(
        num_classes=NUM_CLASSES,
        num_regions=NUM_REGIONS
    )
    model = model.to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 测试评估
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"开始测试...")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试中")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs['final_logits'], labels)

            # 统计指标
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs['final_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "acc": correct / max(total, 1)
            })

    # 计算最终指标
    final_loss = running_loss / len(test_loader.dataset)
    final_acc = correct / total

    print(f"\n📈 测试结果:")
    print(f"测试损失: {final_loss:.4f}")
    print(f"测试准确率: {final_acc:.4f}")
    print(f"最佳历史准确率: {checkpoint.get('best_test_acc', 0.0):.4f}")

    return final_acc