项目简介
本项目是一个基于PyTorch的医学影像多标签分类系统，专为计算机视觉期末课程设计开发。该模型能够对肺部医学影像进行多标签分类，识别包括`Atelectasis`（肺不张）、`Cardiomegaly`（心脏肥大）、`Effusion`（积液）、`Infiltration`（浸润）、`Mass`（肿块）在内的5种常见病变。

模型基于改进的ResNet架构，结合了增强的数据增强策略和类别平衡技术，在光照变化、对比度差异等复杂场景下仍能保持较高的鲁棒性，适合作为医学影像辅助诊断的基础模型。

主要功能
1. 多标签医学影像分类：支持同时识别多种肺部病变，解决医学影像中"一图多病"的标注需求。
2. 增强的数据增强策略：通过随机裁剪、旋转、亮度/对比度调整等技术，提升模型对不同成像条件的适应性。
3. 改进的ResNet模型：引入光照归一化和注意力机制，增强模型对光照变化的鲁棒性。
4. 多标签数据平衡：通过分层抽样和智能过采样技术，解决医学数据中类别不平衡问题。
5. 全面的模型评估：提供精确率、召回率、F1分数、混淆矩阵等多维度评估指标。
6. 鲁棒性测试：自动测试模型在不同亮度/对比度条件下的性能稳定性。
7. 可视化工具：生成训练曲线、预测结果可视化、鲁棒性曲线等分析图表。

环境依赖
- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- numpy 1.19+
- pandas 1.3+
- matplotlib 3.4+
- Pillow 8.0+
- scikit-learn 0.24+
- seaborn 0.11+
- tqdm 4.62+
- skmultilearn 0.2.0+

安装命令：
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn seaborn tqdm scikit-

数据集结构
项目需要以下格式的数据集：
dataset/
├── images/                存放医学影像图片（支持JPG/PNG等格式）
│   ├── img1.jpg
│   ├── img2.jpg
│   ...
└── labels/
    └── all_data.csv       标签文件（CSV格式）

CSV标签文件格式：
- 必须包含`image_path`列：存储图片文件名（如`img1.jpg`）
- 包含5个类别列：`Atelectasis`、`Cardiomegaly`、`Effusion`、`Infiltration`、`Mass`
- 标签值为`0`（无病变）或`1`（有病变）

示例：
image_path,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass
img1.jpg,0,1,0,0,0
img2.jpg,1,0,1,0,0
...

使用方法
1. 准备数据集：
   - 按照上述结构整理数据集
   - 修改代码中`DATA_DIR`、`IMAGE_DIR`、`CSV_PATH`变量为实际路径

2. 运行模型：
   python main.py   假设代码文件名为main.py

3. 主要流程：
   - 数据加载与预处理
   - 自动平衡多标签数据
   - 划分训练集与测试集（分层抽样）
   - 模型训练（含早停机制和混合精度训练）
   - 模型评估与可视化


输出文件
运行完成后，将生成以下文件：
- 模型权重文件：`best_model.pth`（验证集最佳模型）和`robust_resnet_model.pth`（最终模型）
- 可视化结果：
  - `training_history.png`：训练损失和准确率曲线
  - `confusion_matrix.png`：各类别的混淆矩阵
  - `robustness_curve.png`：模型对亮度/对比度变化的鲁棒性曲线
  - `prediction_samples.png`：随机样本的预测结果可视化


 注意事项
1. 数据集路径：请务必在代码中正确设置`DATA_DIR`、`IMAGE_DIR`和`CSV_PATH`变量，确保路径指向实际的图片和标签文件。
2. 硬件要求：训练过程建议使用GPU（代码自动检测CUDA），否则训练速度会显著变慢。
3. 类别修改：若需调整分类类别，需同时修改`CLASSES`列表和标签文件。
4. 参数调整：可根据实际数据量调整`batch_size`、`num_epochs`、`patience`（早停参数）等超参数。
5. 大文件处理：模型权重文件可能超过100MB，若需上传至GitHub，建议使用Git LFS管理（参考`https://git-lfs.github.com`）。

通过本项目，您可以快速搭建一个鲁棒的医学影像多标签分类系统，适合作为计算机视觉课程设计或医学影像分析的入门实践。
