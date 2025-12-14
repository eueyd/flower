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


def load_pretrained_model(model, pretrained_path, freeze_backbone=FREEZE_BACKBONE):
    """加载预训练模型权重"""
    print(f"🔧 加载预训练模型: {pretrained_path}")

    try:
        # 加载预训练权重
        pretrained_state = torch.load(pretrained_path, map_location=DEVICE)

        # 获取预训练模型的state_dict
        if 'model_state_dict' in pretrained_state:
            pretrained_dict = pretrained_state['model_state_dict']
        else:
            pretrained_dict = pretrained_state

        # 当前模型state_dict
        model_dict = model.state_dict()

        # 过滤可以加载的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        # 更新当前模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"✅ 加载了 {len(pretrained_dict)}/{len(model_dict)} 个参数")

        # 冻结骨干网络
        if freeze_backbone:
            freeze_parameters = [
                'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'
            ]

            for name, param in model.named_parameters():
                if any(freeze_name in name for freeze_name in freeze_parameters):
                    param.requires_grad = False
                    print(f"  ❄️ 冻结: {name}")

        return model

    except Exception as e:
        print(f"❌ 加载预训练模型失败: {e}")
        return model


# def train_model_with_pretrain(device, model_path=MODEL_PATH,
#                               metrics_path=METRICS_PATH):
#     """使用预训练模型进行训练"""
#     print(f"\n🚀 开始训练（使用预训练模型）...")
#
#     # 清理缓存
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     # 加载数据
#     from data_loader import create_data_loaders
#     train_loader, test_loader, train_dataset, test_dataset = create_data_loaders()
#
#     print(f"训练集: {len(train_dataset)} 样本")
#     print(f"测试集: {len(test_dataset)} 样本")
#
#     # 构建模型
#     print(f"\n🔧 构建模型...")
#     model = ImprovedDualPathModel(
#         num_classes=NUM_CLASSES,
#         num_regions=NUM_REGIONS,
#         backbone_name=BACKBONE_NAME  # 确保与预训练模型一致
#     )
#     model = model.to(device)
#
#     # 加载预训练权重
#     from config import PRETRAINED_MODEL_PATH, FREEZE_BACKBONE
#     model = load_pretrained_model(model, PRETRAINED_MODEL_PATH, FREEZE_BACKBONE)
#
#     # 统计参数
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     print(f"\n📊 参数统计:")
#     print(f"总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#     print(f"训练比例: {trainable_params / total_params * 100:.1f}%")
#
#     # 分层学习率优化器
#     base_lr = LEARNING_RATE * FINETUNE_RATIO if FREEZE_BACKBONE else LEARNING_RATE
#
#     # 为不同层设置不同学习率
#     param_groups = []
#
#     # 骨干网络参数（如果有未冻结的）
#     backbone_params = []
#     # 新添加的模块参数
#     new_params = []
#
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             if any(module in name for module in [
#                 'region_proposal', 'feature_fusion',
#                 'region_feature_enhancer', 'adaptive_fusion',
#                 'global_classifier', 'final_classifier',
#                 'region_classifiers'
#             ]):
#                 new_params.append(param)
#             else:
#                 backbone_params.append(param)
#
#     if backbone_params:
#         param_groups.append({
#             'params': backbone_params,
#             'lr': base_lr,  # 较低的学习率
#             'weight_decay': 1e-4
#         })
#
#     if new_params:
#         param_groups.append({
#             'params': new_params,
#             'lr': LEARNING_RATE,  # 正常学习率
#             'weight_decay': 1e-3
#         })
#
#     if not param_groups:
#         # 如果所有参数都被冻结，解冻一些层
#         print("⚠️ 所有参数被冻结，解冻部分层...")
#         for name, param in model.named_parameters():
#             if 'final_classifier' in name or 'global_classifier' in name:
#                 param.requires_grad = True
#                 new_params.append(param)
#
#         param_groups.append({
#             'params': new_params,
#             'lr': LEARNING_RATE,
#             'weight_decay': 1e-3
#         })
#
#     optimizer = optim.Adam(param_groups)
#
#     # 学习率调度器
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer,
#         T_0=10,  # 初始周期
#         T_mult=2,  # 周期倍增
#         eta_min=base_lr * 0.01  # 最小学习率
#     )
#
#     # 损失函数
#     criterion = nn.CrossEntropyLoss()
#
#     # 训练指标
#     train_losses, train_accs = [], []
#     test_losses, test_accs = [], []
#     best_test_acc = 0.0
#
#     print(f"\n📈 开始训练，共 {EPOCHS} 个epoch...")
#     print(f"基础学习率: {base_lr:.6f}")
#     print(f"新增模块学习率: {LEARNING_RATE:.6f}")
#
#     for epoch in range(EPOCHS):
#         # ===== 训练阶段 =====
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
#         for batch_idx, (images, labels) in enumerate(pbar):
#             # 定期清理缓存
#             if batch_idx % 20 == 0:
#                 torch.cuda.empty_cache()
#
#             images, labels = images.to(device), labels.to(device)
#
#             # 前向传播
#             outputs = model(images)
#
#             # 计算损失
#             loss, loss_details = model.compute_loss(outputs, labels, criterion)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(
#                 model.parameters(),
#                 max_norm=0.5
#             )
#
#             optimizer.step()
#
#             # 统计
#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs['final_logits'], 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             # 更新进度条
#             pbar.set_postfix({
#                 "loss": running_loss / max(total, 1),
#                 "acc": correct / max(total, 1),
#                 **{k: v for k, v in loss_details.items() if 'loss' in k}
#             })
#
#         train_loss = running_loss / len(train_loader.dataset)
#         train_acc = correct / total
#
#         # ===== 验证阶段 =====
#         model.eval()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         with torch.no_grad():
#             pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
#             for images, labels in pbar:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#
#                 loss = criterion(outputs['final_logits'], labels)
#
#                 running_loss += loss.item() * images.size(0)
#                 _, predicted = torch.max(outputs['final_logits'], 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#
#                 pbar.set_postfix({
#                     "loss": running_loss / max(total, 1),
#                     "acc": correct / max(total, 1)
#                 })
#
#         test_loss = running_loss / len(test_loader.dataset)
#         test_acc = correct / total
#
#         # 学习率调整
#         scheduler.step()
#
#         # 记录指标
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#         test_losses.append(test_loss)
#         test_accs.append(test_acc)
#
#         # 打印结果
#         print(f"\n📊 Epoch {epoch + 1} 结果:")
#         print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
#         print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
#
#         # 逐步解冻策略
#         if epoch == 10 and FREEZE_BACKBONE:  # 10个epoch后解冻部分层
#             print(f"  🔓 逐步解冻layer4...")
#             for name, param in model.named_parameters():
#                 if 'layer4' in name and param.requires_grad == False:
#                     param.requires_grad = True
#                     print(f"    解冻: {name}")
#
#         # 保存最优模型
#         if test_acc > best_test_acc:
#             best_test_acc = test_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'best_test_acc': best_test_acc,
#                 'train_acc': train_acc,
#                 'test_acc': test_acc,
#                 'config': {
#                     'model_type': 'improved_dual_path_pretrained',
#                     'pretrained_path': PRETRAINED_MODEL_PATH,
#                     'freeze_backbone': FREEZE_BACKBONE,
#                     'num_regions': NUM_REGIONS,
#                     'backbone': BACKBONE_NAME,
#                     'num_classes': NUM_CLASSES
#                 }
#             }, model_path)
#             print(f"  💾 保存最优模型，准确率: {best_test_acc:.4f}")
#
#     # 保存训练指标
#     metrics = {
#         'train_losses': train_losses,
#         'train_accs': train_accs,
#         'test_losses': test_losses,
#         'test_accs': test_accs,
#         'best_test_acc': best_test_acc,
#         'epochs': EPOCHS
#     }
#     np.save(metrics_path, metrics)
#
#     print(f"\n✅ 训练完成!")
#     print(f"最佳测试准确率: {best_test_acc:.4f}")
#
#     return model, train_losses, train_accs, test_losses, test_accs


def train_model_with_pretrain(device):
    """修复的预训练模型训练"""
    print(f"\n🚀 使用预训练模型（作为特征提取器）...")

    # 构建简化模型
    from torchvision import models
    import torch.nn as nn

    # 直接使用预训练的ResNet50
    pretrained_model = models.resnet50(pretrained=False)
    num_ftrs = pretrained_model.fc.in_features

    # 加载你的预训练权重
    checkpoint = torch.load("model_89.pth", map_location=device)
    if 'model_state_dict' in checkpoint:
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pretrained_model.load_state_dict(checkpoint)

    print(f"✅ 加载预训练模型，准确率89%")

    # 冻结所有层
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # 只训练最后的分类层（适配你的102类）
    pretrained_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )

    pretrained_model = pretrained_model.to(device)

    # 训练
    optimizer = torch.optim.Adam(pretrained_model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    from data_loader import create_data_loaders
    train_loader, test_loader, _, _ = create_data_loaders()

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(5):  # 少量epoch
        # 训练
        pretrained_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = pretrained_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        pretrained_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = pretrained_model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return pretrained_model, train_losses, train_accs, test_losses, test_accs

def create_exact_resnet50_model(device):
    """创建与model_89.pth完全匹配的ResNet50模型"""
    from torchvision import models
    import torch.nn as nn

    print("🔧 创建完全匹配的ResNet50模型...")

    # 创建标准ResNet50（不要预训练权重）
    model = models.resnet50(weights=None)

    # 修改全连接层以完全匹配你的结构
    print("🔧 修改全连接层以匹配预训练模型...")

    # 你的fc层结构：2048 → 1024 → 512 → 102
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),  # fc.0
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),  # fc.3
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 102)  # fc.6
    )

    # 加载预训练权重
    print("📥 加载预训练权重...")
    checkpoint = torch.load("model_89.pth", map_location=device)

    # 直接加载（应该完全匹配）
    model.load_state_dict(checkpoint, strict=True)

    print("✅ 权重加载成功！完全匹配")

    model = model.to(device)

    # 验证模型
    print("\n🎯 模型验证:")
    print(f"骨干网络: ResNet50")
    print(f"特征维度: 2048")
    print(f"分类器结构: 2048 → 1024 → 512 → 102")

    # 测试前向传播
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(test_input)
        print(f"测试输入: {test_input.shape}")
        print(f"测试输出: {output.shape}")
        print(f"输出维度: {output.shape[1]} (应该是102)")

    return model


def finetune_resnet50(device, freeze_backbone=True):
    """微调ResNet50模型"""
    print(f"\n🚀 开始微调ResNet50模型...")

    # 创建完全匹配的模型
    model = create_exact_resnet50_model(device)

    # 冻结策略
    if freeze_backbone:
        print("\n🔒 冻结骨干网络...")
        frozen_count = 0
        trainable_count = 0

        for name, param in model.named_parameters():
            if 'fc' in name:  # 只训练全连接层
                param.requires_grad = True
                trainable_count += 1
                print(f"  🔓 训练: {name}")
            else:
                param.requires_grad = False
                frozen_count += 1

        print(f"\n📊 冻结统计:")
        print(f"冻结参数: {frozen_count} 个层")
        print(f"可训练参数: {trainable_count} 个层")
    else:
        print("\n🔓 解冻所有层进行训练...")
        for param in model.parameters():
            param.requires_grad = True

    # 加载数据
    from data_loader import create_data_loaders
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders()

    print(f"\n📊 数据统计:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    # 优化器
    if freeze_backbone:
        # 只优化全连接层
        optimizer = torch.optim.Adam(
            model.fc.parameters(),
            lr=0.001,  # 较小学习率
            weight_decay=1e-4
        )
    else:
        # 优化所有参数，但分层学习率
        optimizer = torch.optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 0.0001},
            {'params': model.bn1.parameters(), 'lr': 0.0001},
            {'params': model.layer1.parameters(), 'lr': 0.0001},
            {'params': model.layer2.parameters(), 'lr': 0.0001},
            {'params': model.layer3.parameters(), 'lr': 0.0001},
            {'params': model.layer4.parameters(), 'lr': 0.0001},
            {'params': model.fc.parameters(), 'lr': 0.001}
        ], weight_decay=1e-4)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练循环
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0

    epochs = 20
    print(f"\n📈 开始训练，共 {epochs} 个epoch...")

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {correct / total:.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(test_loader)
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 学习率调整
        scheduler.step()

        print(f"\n📊 Epoch {epoch + 1}/{epochs}:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'train_acc': train_acc,
                'best_acc': best_acc
            }, 'resnet50_finetuned_best.pth')
            print(f"  💾 保存最佳模型，准确率: {best_acc:.4f}")

    print(f"\n✅ 训练完成!")
    print(f"最佳测试准确率: {best_acc:.4f}")

    # 可视化结果
    from visualization import visualize_training_results
    visualize_training_results(
        train_losses, train_accs, test_losses, test_accs,
        save_path='resnet50_finetuning_results.png',
        model_name='ResNet50微调'
    )

    return model, train_losses, train_accs, test_losses, test_accs