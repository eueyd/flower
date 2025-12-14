# config.py
"""
配置参数
"""

# ===================== 模型配置 =====================
MODEL_PATH = "best_flower_finegrained.pth"
METRICS_PATH = "finegrained_metrics.npy"
NUM_CLASSES = 102
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.004  # 直接设为0.04
warmup_factor = 1.0   # 不预热放大
IMAGE_SIZE = (224, 224)
NUM_REGIONS = 2
PRETRAINED_MODEL_PATH = "model_89.pth"  # 添加预训练模型路径
FREEZE_BACKBONE = True  # 是否冻结骨干网络
FINETUNE_RATIO = 0.1    # 微调学习率比例

BACKBONE_NAME = 'resnet50'
# 损失权重
LAMBDA_GLOBAL = 0.1  # 全局损失权重
LAMBDA_REGION = 0.05  # 区域损失权重

# 数据路径
BASE_DIR = "flowers102_data"
TRAIN_DIR = "flowers102_data/train"
TEST_DIR = "flowers102_data/test"

# 类别名称（从flowers102_config导入）
try:
    from flowers102_config import CLASS_NAMES
except ImportError:
    CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]
    print("⚠️  Warning: Using default class names")

# 设备配置
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可视化设置
VISUALIZATION_CONFIG = {
    'style': 'default',
    'figsize': (12, 8),
    'dpi': 100
}