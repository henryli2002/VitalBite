# WABI Food Recognition - Evaluation Framework

This directory contains the scripts and models for evaluating different approaches to food nutrition estimation from images.

---

## 最佳模型

**efficientnet_b0** - 平均 wMAPE 26.4%
- 在 30 样本测试集上表现最优
- 训练轮次: 30 epochs
- 速度: 0.1s/图片

---

## 模型对比 (所有已评估模型)

| 模型 | Mass | Calories | Fat | Carb | Protein | **Avg wMAPE** |
|------|------|----------|-----|------|---------|---------------|
| efficientnet_b0 | 22.5% | 20.8% | 28.1% | 35.9% | 24.5% | **26.4%** ⭐ |
| mobilenet_v3_large | 23.9% | 21.5% | 29.5% | 38.6% | 25.1% | 27.7% |
| mobilenet_v3_small | 27.8% | 25.4% | 33.9% | 42.1% | 30.8% | 32.0% |
| tf_efficientnet_lite4 | 28.5% | 25.8% | 35.2% | 43.1% | 31.9% | 32.9% |
| mobilenetv4_conv_small | 31.1% | 30.2% | 37.7% | 44.2% | 34.6% | 35.6% |
| mobilenetv4_conv_large | 33.9% | 34.1% | 40.2% | 48.8% | 37.1% | 38.8% |

---

## 方案对比 (graph vs direct vs fewshot vs finetuned)

### 准确性 (wMAPE，越低越好)

| 指标 | graph | direct | fewshot | **finetuned** |
|------|-------|--------|---------|---------------|
| Mass | 62.5% | 54.0% | 72.1% | **21.5%** |
| Calories | 135.4% | 79.3% | 51.0% | **21.7%** |
| Fat | 117.9% | 56.2% | 72.5% | **22.2%** |
| Carb | 93.7% | 39.4% | 64.2% | **22.8%** |
| Protein | 102.3% | 52.0% | 71.4% | **30.8%** |

**结论**: finetuned 模型在所有指标上均优于其他方案

### 执行速度

| 方法 | 平均耗时 |
|------|----------|
| finetuned | 0.1s |
| direct | 3.2s |
| fewshot | 4.4s |
| graph | 13.0s |

**结论**: finetuned 比旧流程快 130 倍

---

## 技术选型分析

详见 `doc/finetuning_report.md`

### 为什么不使用目标检测？

1. **任务不匹配**: 目标检测是"识别+定位"，不是"量化"
2. **标注成本高**: 需要精确的边界框标注
3. **难以处理混合食物**: 沙拉、炖菜等无法用方框表示
4. **体积估算困难**: 2D 无法准确推断 3D 重量

### 为什么选择端到端回归？

- 流程简单: 图片 → 营养值
- 标注高效: 仅需图片级别标签
- 学习整体特征
- 速度极快

---

## 使用方法

### 1. 训练模型

```bash
python eval/train_model.py --model efficientnet_b0 --epochs 30
```

### 2. 运行评估

```bash
python eval/run_eval.py --model efficientnet_b0 --n 30
```

### 3. 生成汇总报告

```bash
python eval/summarize_results.py
```

---

## 目录结构

```
eval/
├── models/                    # 所有模型权重
│   ├── efficientnet_b0/       # ⭐ 最佳模型
│   ├── mobilenet_v3_small/
│   ├── mobilenet_v3_large/
│   └── ...
├── food_dataset/             # 训练数据
├── train_model.py            # 训练脚本
├── run_eval.py               # 评估脚本
├── analyze.py                # 分析脚本
└── summarize_results.py      # 汇总脚本
```

---

## 历史记录

原始评估结果归档在 `legacy/eval/` 目录
