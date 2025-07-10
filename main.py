import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_curve, auc
import seaborn as sns
import copy
from tqdm import tqdm
from torch import amp  # 混合精度训练
import sklearn.metrics as metrics
from skmultilearn.model_selection import IterativeStratification

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集路径
DATA_DIR = "D:/python2/视觉课设/dataset"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "labels/all_data.csv")

# 类别映射
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass']
NUM_CLASSES = len(CLASSES)

# 设置 matplotlib 支持中文
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 数据加载和预处理
class MedicalDataset(Dataset):
    """
    医疗图像数据集类，处理图像加载和标签获取
    - 支持图像读取失败时的容错处理
    - 应用数据增强/预处理变换
    """
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 构建完整图像路径
        img_path = self.dataframe.iloc[idx]['image_path']
        img_name = os.path.basename(img_path) if isinstance(img_path, str) else f"default_{idx}.jpg"
        full_img_path = os.path.join(self.image_dir, img_name)

        # 尝试加载图像，失败时创建黑色占位图
        try:
            image = Image.open(full_img_path).convert('RGB')
        except:
            print(f"无法加载图像: {full_img_path}")
            image = Image.new('RGB', (224, 224), color='black')

        # 获取多标签数据并应用变换
        labels = self.dataframe.iloc[idx][CLASSES].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)


# 增强数据增强策略（强化亮度/对比度鲁棒性）
def get_transforms(train=True):
    """
    生成数据变换流水线
    - 训练集：包含随机增强（裁剪、翻转、旋转等）
    - 测试集：仅基础预处理（中心裁剪、归一化）
    """
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),          # 随机裁剪增强空间鲁棒性
            transforms.RandomHorizontalFlip(),   # 水平翻转增强
            transforms.RandomRotation(15),       # 随机旋转增强
            transforms.ToTensor(),
            transforms.ColorJitter(contrast=0.5), # 对比度增强
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # 锐度增强
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # 高斯模糊增强
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)), # 随机擦除增强
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 归一化
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),  # 测试时使用中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# 改进的ResNet模型（增强光照鲁棒性）
class RobustResNet(nn.Module):
    """
    改进的ResNet50模型，增强光照鲁棒性：
    1. 解冻中层和高层特征提取层
    2. 添加光照归一化和注意力机制
    3. 改进分类头（多层FC+Dropout）
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(RobustResNet, self).__init__()
        # 加载预训练ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # 解冻中层和高层特征提取层（仅训练这些层）
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # 改进分类层：添加多层FC和高dropout率抑制过拟合
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),  # 高dropout率增强泛化能力
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

        # 增强光照鲁棒性的模块
        self.light_norm = nn.InstanceNorm2d(3, affine=True)  # 实例归一化增强光照不变性
        self.illumination_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 2048, kernel_size=1),
            nn.Sigmoid()
        )  # 光照注意力机制，强化光照不变特征

        # 新增亮度/对比度不变性模块
        self.light_inv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        )

    def forward(self, x):
        # 光照归一化预处理
        x = self.light_norm(x)
        x = self.light_inv(x)  # 亮度/对比度不变性处理

        # 标准ResNet特征提取流程
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # 光照注意力：对特征图加权，强化光照不变特征
        att = self.illumination_att(x4)
        x4 = x4 * att  # 注意力加权

        # 全局平均池化和分类
        x = self.resnet.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


# 优化的多标签数据平衡
def balance_multilabel_data(df, classes=CLASSES, target_size=1000):
    """
    智能过采样：对少数类进行过采样，多数类保持原样
    - 优先选择未使用的样本，减少重复采样
    - 确保每个类别的样本数量接近target_size
    """
    balanced_dfs = []
    all_indices = set()  # 跟踪已选索引避免重复

    for cls in classes:
        cls_df = df[df[cls] == 1]  # 筛选当前类别的正例
        sample_size = min(target_size, len(cls_df))  # 采样数量上限

        # 优先选择未被其他类别采样过的样本
        available_df = cls_df[~cls_df.index.isin(all_indices)]
        if len(available_df) < sample_size:
            # 若可用样本不足，补充已使用样本（带替换采样）
            sampled_df = pd.concat([
                available_df,
                cls_df.sample(n=sample_size - len(available_df), replace=True, random_state=42)
            ])
        else:
            sampled_df = available_df.sample(n=sample_size, random_state=42)

        balanced_dfs.append(sampled_df)
        all_indices.update(sampled_df.index)  # 更新已使用索引

    # 合并所有类别并打乱顺序
    balanced_df = pd.concat(balanced_dfs, ignore_index=True).drop_duplicates().sample(frac=1, random_state=42)
    return balanced_df


# 多标签分层抽样
def multilabel_stratified_split(df, test_size=0.2, random_state=42):
    """
    基于IterativeStratification的多标签分层抽样
    - 尝试使用多标签分层方法，失败时回退到单标签分层
    - 确保训练集和测试集的类别分布相似
    """
    np.random.seed(random_state)
    X = df.index.values.reshape(-1, 1)  # 特征（仅索引）
    y = df[CLASSES].values  # 多标签

    # 修补类以解决兼容性问题
    class PatchedIterativeStratification(IterativeStratification):
        def __init__(self, n_splits=2, order=1, sample_distribution_per_fold=None):
            super().__init__(n_splits=n_splits, order=order, sample_distribution_per_fold=sample_distribution_per_fold)

    try:
        # 尝试使用多标签分层抽样
        stratifier = PatchedIterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[test_size, 1.0 - test_size]
        )
        train_indices, test_indices = next(stratifier.split(X, y))
        assert not set(train_indices).intersection(set(test_indices)), "训练集与测试集索引重叠"

    except Exception as e:
        print(f"多标签分层抽样失败: {e}")
        # 回退到基于部分类别的单标签分层
        from sklearn.model_selection import train_test_split
        stratify_cols = ['Atelectasis', 'Effusion', 'Mass']
        train_indices, test_indices = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_cols].apply(tuple, axis=1)
        )

    # 创建训练集和测试集
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # 输出划分统计信息
    actual_test_size = len(test_df) / len(df)
    print(f"目标测试集比例: {test_size:.2f}, 实际测试集比例: {actual_test_size:.2f}")
    print("测试集类别分布:")
    print(test_df[CLASSES].sum())

    return train_df, test_df


# 计算类别权重
def calculate_class_weights(df, classes=CLASSES):
    """
    根据各类别样本数动态计算权重
    - 权重公式: (总样本数 + 平滑项) / (正例数 + 平滑项)
    - 平衡类别不平衡问题
    """
    pos_counts = df[classes].sum()  # 各类别正例数
    total_samples = len(df)  # 总样本数
    smooth = 1e-5  # 平滑项，避免除零
    class_weights = torch.FloatTensor([(total_samples + smooth) / (pos_counts[cls] + smooth) for cls in classes]).to(
        device)
    print(f"类别权重: {class_weights}")
    return class_weights


# 训练函数（含早停和混合精度训练）
def train_model(model, criterion, optimizer, scheduler, num_epochs=30, patience=5):
    """
    训练模型并应用早停机制和混合精度训练
    - 记录训练和验证损失/准确率
    - 保存验证集表现最佳的模型
    - 使用混合精度训练加速计算
    """
    best_model = copy.deepcopy(model.state_dict())  # 保存最佳模型权重
    best_val_acc = 0.0  # 最佳验证准确率
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}  # 训练历史
    no_improvement = 0  # 早停计数器

    # 混合精度训练设置
    scaler = amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"训练Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 混合精度前向传播
            with amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 计算准确率
                preds = torch.sigmoid(outputs) > 0.5
                correct = (preds == labels.byte()).sum().item()
                running_corrects += correct
                total_samples += labels.numel()

            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)

        # 计算训练损失和准确率
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"验证Epoch {epoch + 1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5

                correct = (preds == labels.byte()).sum().item()
                val_corrects += correct
                val_total += labels.numel()
                val_loss += loss.item() * inputs.size(0)

        # 计算验证损失和准确率
        val_loss = val_loss / len(test_dataset)
        val_acc = val_corrects / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印训练日志
        print(f"Epoch {epoch + 1} - 训练损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}")
        print(f"验证损失: {val_loss:.4f} 准确率: {val_acc:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.7f}")

        # 学习率调整
        scheduler.step(val_acc)

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, 'best_model.pth')  # 保存最佳模型
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"早停触发：连续{patience}个Epoch验证集表现未提升")
                break

    # 加载最佳模型
    model.load_state_dict(best_model)
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return model, history


# 绘制学习曲线
def plot_training_history(history):
    """
    绘制训练和验证损失/准确率曲线
    - 保存图像为 training_history.png
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# 模型评估
def evaluate_model(model, test_loader):
    """
    全面评估模型性能：
    1. 各类别的分类报告（精确率、召回率、F1）
    2. 多标签总体指标（Micro/Macro）
    3. 混淆矩阵可视化
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    # 分类报告（各类别）
    print("\n各类别分类报告:")
    for i, cls in enumerate(CLASSES):
        print(f"\n--- {cls} ---")
        report = classification_report(all_labels[:, i], all_preds[:, i],
                                       target_names=['负例', '正例'], zero_division=1)
        print(report)

    # 总体指标
    print("\n多标签总体指标:")
    print(f"Micro Precision: {metrics.precision_score(all_labels, all_preds, average='micro'):.4f}")
    print(f"Macro Precision: {metrics.precision_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Micro Recall: {metrics.recall_score(all_labels, all_preds, average='micro'):.4f}")
    print(f"Macro Recall: {metrics.recall_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Micro F1: {metrics.f1_score(all_labels, all_preds, average='micro'):.4f}")
    print(f"Macro F1: {metrics.f1_score(all_labels, all_preds, average='macro'):.4f}")

    # 混淆矩阵
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 10))
    for i, (cls, matrix) in enumerate(zip(CLASSES, mcm)):
        plt.subplot(2, 3, i + 1)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测负例', '预测正例'],
                    yticklabels=['真实负例', '真实正例'])
        plt.title(f'混淆矩阵: {cls}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return all_preds, all_labels, all_probs


# 鲁棒性测试（亮度/对比度变化）
def test_robustness(model, test_df, batch_size=16):
    """
    测试模型对亮度/对比度变化的鲁棒性：
    - 在不同亮度/对比度强度下评估模型
    - 绘制鲁棒性曲线
    """
    model.eval()
    robustness_results = []
    # 扩展测试范围，增加中间点
    intensity_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    print("开始鲁棒性测试...")
    for intensity in intensity_range:
        # 创建特定亮度/对比度的变换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=intensity, contrast=intensity),  # 关键测试点
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 创建测试数据集和加载器
        robust_dataset = MedicalDataset(test_df, IMAGE_DIR, transform=transform)
        robust_loader = DataLoader(robust_dataset, batch_size=batch_size, shuffle=False)

        # 评估模型
        corrects, total = 0, 0
        with torch.no_grad():
            for inputs, labels in robust_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5
                corrects += (preds == labels.byte()).sum().item()
                total += labels.numel()

        # 计算准确率并记录结果
        acc = corrects / total
        robustness_results.append(acc)
        print(f"亮度/对比度强度 {intensity:.1f} - 准确率: {acc:.4f}")

    # 绘制鲁棒性曲线
    plt.figure(figsize=(10, 6))
    plt.plot(intensity_range, robustness_results, 'o-', markersize=8)
    plt.title('模型对亮度/对比度变化的鲁棒性')
    plt.xlabel('变化强度'), plt.ylabel('准确率')
    plt.grid(True), plt.ylim(0, 1)
    for i, txt in enumerate(robustness_results):
        plt.annotate(f'{txt:.4f}', (intensity_range[i], robustness_results[i]), xytext=(0, 10),
                     textcoords='offset points', ha='center')
    plt.savefig('robustness_curve.png')
    plt.show()

    return robustness_results


# 可视化预测结果
def visualize_predictions(model, test_df, num_samples=6):
    """
    可视化模型预测结果：
    - 显示原始图像和真实标签
    - 显示预测标签和概率分布
    """
    model.eval()
    # 随机选择样本
    sample_df = test_df.sample(num_samples)

    transform = get_transforms(train=False)  # 使用测试集变换
    plt.figure(figsize=(10, 2 * num_samples))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        # 加载图像
        img_path = os.path.join(IMAGE_DIR, os.path.basename(row['image_path']))
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')

        # 模型预测
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # 获取真实标签和预测标签
        true_labels = [CLASSES[j] for j in range(NUM_CLASSES) if row[CLASSES[j]] == 1]
        pred_labels = [CLASSES[j] for j in range(NUM_CLASSES) if probs[j] > 0.5]

        # 绘制图像和预测结果
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(image), plt.axis('off')
        plt.title(f"真实: {', '.join(true_labels)}\n预测: {', '.join(pred_labels)}", fontsize=8)

        # 绘制概率分布
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.barh(CLASSES, probs, color='skyblue')
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.title('概率分布', fontsize=8)

    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()


# 主函数
if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv(CSV_PATH)
    print(f"原始总样本数: {len(df)}")
    print("原始类别分布:")
    print(df[CLASSES].sum())
    df.fillna(0, inplace=True)  # 填充缺失值

    # 数据平衡
    balanced_df = balance_multilabel_data(df)
    print(f"过采样后总样本数: {len(balanced_df)}")

    # 分层抽样
    train_df, test_df = multilabel_stratified_split(balanced_df, test_size=0.2, random_state=42)

    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    print("训练集类别分布:")
    print(train_df[CLASSES].sum())
    print("测试集类别分布:")
    print(test_df[CLASSES].sum())

    # 验证训练集与测试集无重叠
    assert not train_df.index.isin(test_df.index).any(), "错误：训练集与测试集存在数据重叠"

    # 创建数据集和数据加载器
    train_dataset = MedicalDataset(train_df, IMAGE_DIR, transform=get_transforms(train=True))
    test_dataset = MedicalDataset(test_df, IMAGE_DIR, transform=get_transforms(train=False))

    # 数据加载器配置
    batch_size = 16
    train_loader = DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )

    print(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")

    # 初始化模型
    model = RobustResNet().to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params / 1e6:.2f}M, 可训练参数: {trainable_params / 1e6:.2f}M")

    # 动态计算类别权重
    class_weights = calculate_class_weights(balanced_df)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # 带类别权重的损失函数

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # 训练模型
    num_epochs = 30
    model, history = train_model(model, criterion, optimizer, scheduler, num_epochs, patience=5)

    # 评估与可视化
    plot_training_history(history)  # 绘制训练历史
    evaluate_model(model, test_loader)  # 评估模型性能
    robustness_results = test_robustness(model, test_df)  # 鲁棒性测试
    visualize_predictions(model, test_df)  # 可视化预测结果

    # 保存模型
    torch.save(model.state_dict(), 'robust_resnet_model.pth')
    print("模型已保存至 robust_resnet_model.pth")
