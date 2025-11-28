# 颈动脉斑块分割与分类项目

## 项目简介

本项目实现了基于半监督学习的颈动脉斑块分割与分类系统，采用Mean Teacher框架对颈动脉超声图像（长轴和横轴）进行多任务学习：
- **分割任务**：识别血管和斑块区域（3类分割：背景、血管、斑块）
- **分类任务**：判断斑块风险等级（RADS2 vs RADS3-4，二分类）

## 项目结构

```
release_version/
├── baseline/                          # 基线模型训练代码
│   ├── Dockerfile                    # Docker镜像配置
│   ├── Dataset.py                    # 数据集加载与数据增强
│   ├── model.py                       # 模型定义
│   ├── train.py  # 半监督训练脚本
│   ├── test.py                       # 测试评估脚本
│   ├── augmentations/                # 数据增强模块
│   │   ├── __init__.py
│   │   └── ctaugment.py             # CTAugment增强策略
│   └── utils/                        # 工具函数
│       ├── eval_utils.py            # 评估指标计算（Dice、NSD等）
│       ├── losses.py                # 损失函数
│       ├── ramps.py                 # 一致性权重调度
│       ├── util.py                  # 通用工具
│       └── vis_utils.py             # 可视化工具
│
├── docker_submission_template/       # Docker提交模板
│   ├── run_infer.py                 # 推理脚本
│   ├── Dockerfile                   # Docker镜像配置
│   ├── requirements.txt             # Python依赖
│   ├── models/                      # 模型代码目录（需放置模型文件）
│   └── weights/                     # 模型权重目录（需放置.pth文件）
│
└── Readme.md                         # 项目说明文档
```


## 使用指南

### 1. 获取项目文件
```bash
git clone https://github.com/xxx/xxx.git
```
文件中包含baseline和docker_submission_template两个docker镜像，分别是基线模型和提交模板。

### 2. 训练模型

#### 数据准备
数据应按以下结构组织：
```
data/
└── train/
    ├── train.list          # 训练样本列表
    ├── img/                # 图像文件（.h5格式）
    │   ├── case_001.h5    # 包含 long_img 和 trans_img 数据集
    │   └── ...
    └── mask/               # 标注文件（.h5格式）
        ├── case_001_label.h5  # 包含 long_mask, trans_mask, cls 数据集
        └── ...

```

#### 训练参数配置
编辑 `baseline/train.py` 中的 `Args` 类：
```python
class Args:
    root_path = "/path/to/data"           # 数据路径
    max_iterations = 30000                 # 最大迭代次数
    batch_size = 24                        # 总批次大小
    labeled_bs = 8                         # 有标签样本批次大小
    num_labeled = 200                      # 有标签样本数量
    base_lr = 0.01                         # 初始学习率
    ema_decay = 0.99                       # EMA衰减率
    seg_consistency = 0.1                  # 分割一致性权重
    cls_consistency = 0.1                  # 分类一致性权重
```

#### 基线模型训练
```bash
python train.py
```

训练过程会：
- 每250次迭代打印训练日志
- 每500次迭代进行验证评估
- 每3000次迭代保存检查点
- 自动保存最佳Dice模型和最佳分类Score模型
- 在 `result/experiment_name/` 目录下保存日志和TensorBoard记录

### 3. 模型测试

#### 配置测试参数
编辑 `baseline/test.py` 中的 `Args` 类：
```python
class Args:
    root_path = "/path/to/data"           # 测试数据路径
    model_path = "/path/to/model.pth"     # 模型权重路径
    batch_size = 1                         # 测试批次大小
    save_vis = "/path/to/save/results"    # 结果保存路径
    num_vis_cases = 5                      # 可视化病例数
```

#### 运行测试
```bash
python test.py
```

### 4. About model.py ⭐

**`model.py` is the core file!** You can freely modify its internal implementation, but must follow these specifications:

#### Required Interface

```python
class Model(nn.Module):
    def __init__(self, in_chns=1, seg_classes=2, cls_classes=2):
        """Initialize model"""
        pass
    
    def forward(self, x_long, x_trans):
        """
        Perform inference
        
        Args:
            data_root: Input data root directory (/input/ in Docker)
            output_dir: Output directory (/output/ in Docker)
            batch_size: Batch size
        """
        pass
```

### 4. About run_infer.py ⭐

**`run_infer.py` is the core file for testing.** Please do not modify its parameters, otherwise the test procedure will fail.




## 提交方式

#### 提交结果
- 将模型文件命名为`model.py`，并放入`docker_submission_template/models/`目录下
- 将权重文件命名为`best_model.pth`，并放入`docker_submission_template/weights/`目录下

首先请训练好的模型权重和测试脚本按照上述结构组织成提交文件，然后按照以下指示提交：

#### Method A: Docker Hub (Recommended)

```bash
docker login
docker tag my-submission:latest YOUR_USERNAME/my-submission:latest
docker push YOUR_USERNAME/my-submission:latest

# Send the image address to the organizing committee:
# YOUR_USERNAME/my-submission:latest
```

#### Method B: Save as File

```bash
docker save -o my-submission.tar my-submission:latest

# Upload my-submission.tar to cloud storage
# Send the link to the organizing committee
```

我们提供了三种提交路径：
- **Docker提交 (适用Method A)**：将Docker镜像上传到Docker Hub项目，然后将链接发送到邮箱
- **Google Drive (适用Method B)**：将Docker镜像上传至Google Drive，然后通过链接提交。
- **百度网盘 (适用Method B)**：将Docker镜像上传至百度网盘，然后通过链接提交。
 
[EMAIL LINK](zhuzhiyuan113@163.com)

## FAQ

### Q1: Docker image push failed frequently?

**Answer**: 
- In China mainland, you may need to use a proxy server to access Docker Hub.
- You can submit your image via Google Drive or Baidu Netdisk.