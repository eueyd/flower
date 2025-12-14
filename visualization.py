# visualization.py
"""
可视化函数
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl

from config import CLASS_NAMES, VISUALIZATION_CONFIG


def setup_chinese_font():
    """设置中文字体支持"""
    available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]

    chinese_fonts = [
        'WenQuanYi Micro Hei',
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'STHeiti',
        'STKaiti',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
    ]

    for font in chinese_fonts:
        if font in available_fonts:
            try:
                plt.rcParams['font.family'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 中文字体已设置为: {font}")
                return True
            except:
                continue

    print("⚠️  警告：未找到合适的中文字体，将使用默认字体")
    return False


def set_plot_style(style='default', figsize=(12, 8), dpi=100):
    """设置图表样式"""
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi

    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
               '#bcbd22', '#17becf']
    )

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.linewidth'] = 1.2

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'seaborn':
        plt.style.use('seaborn-v0_8')
    elif style == 'ggplot':
        plt.style.use('ggplot')

    return True


def visualize_training_results(train_losses, train_accs, test_losses, test_accs,
                               save_path='training_results.png', model_name='双路径模型'):
    """可视化训练结果"""
    setup_chinese_font()
    set_plot_style(style=VISUALIZATION_CONFIG['style'],
                   figsize=VISUALIZATION_CONFIG['figsize'],
                   dpi=VISUALIZATION_CONFIG['dpi'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    axes[0].plot(epochs, test_losses, 'r-', linewidth=2, label='测试损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title(f'{model_name}损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制准确率曲线
    axes[1].plot(epochs, train_accs, 'g-', linewidth=2, label='训练准确率')
    axes[1].plot(epochs, test_accs, 'orange', linewidth=2, label='测试准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title(f'{model_name}准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 训练结果已保存为: {save_path}")
    return save_path


def visualize_sample_attention(model, test_loader, device, num_samples=3):
    """可视化样本注意力图"""
    setup_chinese_font()
    set_plot_style(figsize=(15, 3 * num_samples), dpi=100)

    # 获取数据
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images[:num_samples].to(device)
        labels = labels[:num_samples]

        outputs, attention_maps = model(images, return_attention=True)

    # 创建图表
    from config import NUM_REGIONS

    fig, axes = plt.subplots(
        num_samples, NUM_REGIONS + 2,
        figsize=(15, 3 * num_samples)
    )

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # 原始图像
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"原始\n{CLASS_NAMES[labels[i]]}")
        axes[i, 0].axis('off')

        # 注意力图
        for j in range(NUM_REGIONS):
            att_map = attention_maps[i, j].cpu().numpy()
            axes[i, j + 1].imshow(att_map, cmap='hot')
            weight = outputs['fusion_weights'][i, j + 1].item() if 'fusion_weights' in outputs else 0.0
            axes[i, j + 1].set_title(f"区域{j + 1}\nw={weight:.2f}")
            axes[i, j + 1].axis('off')

        # 融合权重
        if 'fusion_weights' in outputs:
            weights = outputs['fusion_weights'][i].cpu().numpy()
            colors = ['blue'] + ['green', 'red'][:len(weights) - 1]
            axes[i, -1].bar(range(len(weights)), weights, color=colors)
            axes[i, -1].set_title("融合权重")
            axes[i, -1].set_xticks(range(len(weights)))
            axes[i, -1].set_xticklabels(['全局'] + [f'局部{i + 1}' for i in range(len(weights) - 1)])
            axes[i, -1].set_ylim(0, 1.0)

    plt.suptitle('双路径模型注意力可视化', fontsize=14)
    plt.tight_layout()

    save_path = 'attention_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 注意力可视化已保存为: {save_path}")
    return fig