# 自训练模型来源与训练记录

2026-11-06：按归档要求，删除下述本地训练权重、中间 checkpoint 和配套数据，仅保留训练代码、配置、来源说明和文本评测报告。uv 恢复软件依赖，不会恢复权重或数据。删除文件的路径、大小、SHA-256 见 training-cleanup-manifest.json；该文件只含元信息。

## 意图路由 LoRA

- 实际保存的 adapter 配置指向本地基础模型 `~/.omlx/models/Qwen3.5-0.8B-OptiQ-4bit`。仓库未记录此量化版本的 Hugging Face repo/revision，不能据名称推断精确上游来源。当前生产路由使用 Qwen3.5-9B-OptiQ-4bit 零样本，与被删除的 0.8B LoRA 不同；legacy 外的 ~/.omlx 模型未处理。
- 训练方式：MLX-LM LoRA，配置保留在 `eval/intent/train/adapters/*/adapter_config.json` 与 `eval/intent/train/experiments/*.yaml`。`run_sweep.sh` 使用 oMLX 内置 Python 3.11 和 gradient checkpointing，迁移时需修改机器绝对路径。
- A：最后 8 层全部线性模块；B：最后 8 层 attention（q_proj/v_proj/o_proj、linear_attn in_proj_qkv/out_proj）；C：最后 8 层 MLP（up_proj/gate_proj/down_proj）；D：最后 16 层全部线性模块。共有 rank=8、scale=16、dropout=0、500 iterations、lr=2e-4、batch=1、梯度累积=4、序列长 512、seed=42，每 100 步保存、50 步验证。最终 adapter 与 100/200/300/400/500 步 checkpoint 均删除。qwen-0.8b-r8 仅存的配置也保留（batch=4、序列长 2048），未发现其权重。
- 数据来源：`eval/intent/gen_dataset.py` 中定义的中英双语合成/人工编写意图样本；包含 recognition / recommendation / chitchat / goalplanning、normal 与 abnormal 等类别。生成脚本保留，可重新生成样本文件。
- 划分：normal 300 + abnormal 100；`split_dataset.py --seed 42` 产生 train 300、test 100；`prepare_mlx.py` 对训练集 shuffle(seed=42)，90/10 分成 MLX train 270、valid 30，转换为 system/history/user/assistant messages。assistant 目标含 INTENT、CONFIDENCE、REASONING；默认注入由 gen_profiles 生成的用户画像，图片用 UUID 文本标记。
- 已删除：dataset 下生成的 normal/abnormal/train/test/smoke JSONL，mlx_train/mlx_valid 及 train/valid 符号链接，fixtures 中测试图片。来源脚本和已汇总的文本评测结论保留；重新评测需重建数据及图片 fixture。
- 历史 `train_lora.sh` 指向 gemma-4-e4b-it-4bit（batch=4、8 层、rank=8、scale=20、600 steps、lr=1e-4），是另一份训练方案，不应据此把现有 Qwen adapter 误记为 Gemma 产物。

## 食物营养回归模型

- 骨干来源：torchvision 的 DEFAULT 预训练 MobileNet V3 Small/Large、EfficientNet-B0；timm 的 pretrained MobileNet V4 Conv Small/Large、TF EfficientNet Lite4。仓库未记录预训练权重的历史版本/哈希。
- 训练入口：`eval/food_recognition/train_model.py`；torchvision 分支回归头为输入特征→256→ReLU→Dropout(0.3)→5；timm 分支 num_classes=5。5 个目标为 total_mass、total_calories、total_fat、total_carb、total_protein。
- 数据：`eval/food_recognition/food_dataset/dish_images.pkl`（2,501,055,235 bytes，含图像表，代码读取 dish / rgb_image）和 `dishes.xlsx`（233,725 bytes，单 sheet 共 5006 条非表头行）。按 dish 与 dish_id inner join。仓库未找到数据集原始名称、下载地址、版本或许可证，不能保证重新取得相同数据；未反序列化 pickle。
- 划分/训练默认：seed=42 打乱 dish ID、留出前 30 个测试 ID；其余样本 shuffle 后 90/10 分 train/val。30 epochs、lr=1e-3、batch=32、AdamW(weight_decay=1e-4)、CosineAnnealingLR(T_max=epochs)、SmoothL1Loss；以最小 val loss 保存 best，末轮保存 final。实际每次运行的所有 CLI 参数未记录，以上是当前脚本默认与现有 README 的记录。
- 输入：训练 Resize 256×256→RandomCrop 224→随机翻转→ColorJitter(0.2,0.2,0.1)；验证 224×224；ImageNet mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]；目标按训练集 mean/std 标准化。norm_stats.json 保留，属于参数元信息。
- 6 组实验目录的 best/final 权重以及运行目录的 EfficientNet-B0 best_model.pt 均删除；删除前已验证实验 EfficientNet best 与运行副本 SHA-256 一致。旧文档中的 MobileNet 结论与当前 EfficientNet 部署属于不同阶段，保留原报告但不混同。
- 重建：先重新取得/导出上述 schema 数据，再运行 `python eval/food_recognition/train_model.py --model efficientnet_b0 --epochs 30`。当前脚本输出在 `eval/food_recognition/model_<name>/`，历史文件整理在 `eval/food_recognition/models/model_<name>/`；部署时将重新训练的 best 与配套 norm_stats 复制到 `src/langgraph_app/agents/food_recognition/model/`。

## 删除后的功能状态

营养预测代码仍保留，但本地 best_model.pt 已不存在，直接调用会报缺少模型；必须先恢复数据并重新训练。LoRA 扫描评测亦需重新训练。数据库、用户上传图片、其他项目语料和外部基础模型未纳入本轮训练产物清理。Git 历史不改写；删除不会擦除历史提交中的文件。
