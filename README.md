# seq-model

事件序列二分类项目，包含训练入口与基准分析笔记本。

## 推荐结构

- `train.py`: 项目级训练入口（自动数据处理、切分、训练、导出）
- `benchmark.ipynb`: 加载训练产物并做离线分析
- `event_classifier_v2/`: 模型、数据集、训练循环、推理及流水线工具
- `dataset/`: 原始数据目录（默认不入库）
- `artifacts/`: 训练产物目录（默认不入库）
  - `models/best_model.pt`
  - `predictions/{train,val,test}_predictions.csv`
  - `run_metadata.json`

## 训练

```bash
python train.py \
  --seq-data-path ./dataset/event_lst.csv \
  --sample-path ./dataset/tmp_0403.csv \
  --output-dir ./artifacts
```

## 产物说明

训练完成后会输出：

- 模型文件：`artifacts/models/best_model.pt`
- 评估元数据：`artifacts/run_metadata.json`
- 预测明细：
  - `artifacts/predictions/train_predictions.csv`
  - `artifacts/predictions/val_predictions.csv`
  - `artifacts/predictions/test_predictions.csv`

预测 CSV 至少包含：

- `sn`
- `verify_time`
- `target`
- `split`
- `prob`
- `score`

## 分析

打开 `benchmark.ipynb` 直接运行，即可查看：

- train/val/test 的 AUC、PR-AUC、KS
- test 集分箱统计
- test 集 score 分布与分箱坏账率曲线
