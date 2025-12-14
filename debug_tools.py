# debug_tools.py
"""
调试工具模块
"""

import torch
import torch.nn as nn
import numpy as np


class ModelDebugger:
    """模型调试工具"""

    @staticmethod
    def check_module_output(module, input_tensor, module_name):
        """检查模块输出"""
        print(f"\n🔍 检查 {module_name}:")
        print(f"  输入形状: {input_tensor.shape}")

        try:
            with torch.no_grad():
                output = module(input_tensor)

            print(f"  输出形状: {output.shape}")

            if torch.is_tensor(output):
                print(f"  数值范围: [{output.min():.6f}, {output.max():.6f}]")
                print(f"  均值: {output.mean():.6f}, 标准差: {output.std():.6f}")

                if torch.isnan(output).any():
                    print("  ⚠️ 警告: 输出包含NaN!")
                if torch.isinf(output).any():
                    print("  ⚠️ 警告: 输出包含Inf!")

            return output

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return None

    @staticmethod
    def check_gradient(model, loss):
        """检查梯度"""
        print(f"\n🔍 检查梯度:")

        total_grad_norm = 0
        zero_grad_params = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm

                if grad_norm < 1e-6:
                    zero_grad_params.append((name, grad_norm))
                elif torch.isnan(param.grad).any():
                    print(f"  ❌ {name}: 梯度包含NaN!")
            else:
                print(f"  ⚠️ {name}: 梯度为None")

        print(f"  总梯度范数: {total_grad_norm:.6f}")

        if zero_grad_params:
            print(f"  零梯度参数 ({len(zero_grad_params)}个):")
            for name, norm in zero_grad_params[:5]:  # 只显示前5个
                print(f"    {name}: {norm:.6e}")
            if len(zero_grad_params) > 5:
                print(f"    ... 还有 {len(zero_grad_params) - 5} 个")

    @staticmethod
    def check_attention_maps(attention_maps):
        """检查注意力图"""
        if attention_maps is None:
            print("  注意力图为None")
            return

        print(f"\n🔍 检查注意力图:")
        print(f"  形状: {attention_maps.shape}")
        print(f"  数值范围: [{attention_maps.min():.6f}, {attention_maps.max():.6f}]")

        # 检查每个区域的注意力总和
        if attention_maps.dim() == 4:
            print(f"  每个像素的注意力值总和:")
            for i in range(attention_maps.shape[1]):
                region_sum = attention_maps[:, i:i + 1].sum(dim=1).mean().item()
                print(f"    区域{i + 1}: {region_sum:.6f}")

    @staticmethod
    def check_model_parameters(model):
        """检查模型参数"""
        print(f"\n🔍 模型参数统计:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  冻结参数: {total_params - trainable_params:,}")

        return total_params, trainable_params

    @staticmethod
    def test_forward_pass(model, input_shape=(2, 3, 224, 224), device='cpu'):
        """测试前向传播"""
        print(f"\n🔍 测试前向传播:")

        dummy_input = torch.randn(*input_shape).to(device)
        model.eval()

        try:
            with torch.no_grad():
                outputs = model(dummy_input)

            print(f"  ✅ 前向传播成功!")

            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if torch.is_tensor(value):
                        print(f"    {key}: {value.shape}")
                    elif value is not None:
                        print(f"    {key}: {type(value).__name__}")

            return outputs

        except Exception as e:
            print(f"  ❌ 前向传播失败: {e}")
            return None


def test_basic_functionality(device):
    """测试模型基本功能"""
    print("=" * 60)
    print("🧪 基础功能测试")
    print("=" * 60)

    debugger = ModelDebugger()

    # 测试输入
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    print(f"\n📊 测试输入:")
    print(f"  形状: {dummy_input.shape}")
    print(f"  范围: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")

    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(64 * 112 * 112, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleModel().to(device)

    # 测试前向传播
    print(f"\n🔧 测试简单模型:")
    debugger.test_forward_pass(model, input_shape=(2, 3, 224, 224), device=device)

    # 测试梯度
    print(f"\n📈 测试梯度计算:")
    model.train()
    output = model(dummy_input)
    dummy_labels = torch.randint(0, 10, (2,)).to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_labels)
    loss.backward()
    debugger.check_gradient(model, loss)

    print("\n" + "=" * 60)
    print("✅ 基础功能测试完成")
    print("=" * 60)

    return True


# 添加一个新的检查函数到debug_tools.py
def check_pretrained_model(model_path):
    """检查预训练模型"""
    print(f"\n🔍 检查预训练模型: {model_path}")

    if not os.path.exists(model_path):
        print("❌ 模型文件不存在")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        print("✅ 模型加载成功")
        print(f"文件大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"参数数量: {len(state_dict)}")

            # 打印前几个键
            print("\n参数键示例:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i + 1}. {key}: {state_dict[key].shape}")

            # 检查是否有分类器
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k]
            if classifier_keys:
                print(f"\n分类器层 ({len(classifier_keys)}个):")
                for key in classifier_keys:
                    print(f"  {key}: {state_dict[key].shape}")

        if 'config' in checkpoint:
            print(f"\n模型配置:")
            for key, value in checkpoint['config'].items():
                print(f"  {key}: {value}")

        if 'best_test_acc' in checkpoint:
            print(f"\n历史最佳准确率: {checkpoint['best_test_acc']:.4f}")

        return checkpoint

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return None