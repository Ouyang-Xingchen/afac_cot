# 金融领域长思维链压缩基线方法

本项目实现了一个基于 Qwen3-4B 模型的金融领域长思维链压缩基线方法。该方法通过优化思维链的表达方式，在保持答案准确性的同时，显著减少思维链的长度。

## 环境要求

- Python 3.8+
- CUDA 11.7+ (用于 GPU 加速)
- 至少 16GB GPU 显存 (用于运行 Qwen3-4B 模型)

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 创建并激活虚拟环境（推荐使用 conda）：
```bash
conda create -n afac_cot python=3.8
conda activate afac_cot
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
.
├── example_baseline/
│   ├── data/                    # 数据目录
│   │   ├── input.csv           # 输入数据
│   │   └── reference.csv       # 参考答案
│   ├── main.py                 # 主程序
│   ├── data_loader.py          # 数据加载模块
│   ├── model_inference.py      # 模型推理模块
│   ├── evaluator.py            # 评估模块
│   └── generate_reference.py   # 生成参考答案脚本
├── requirements.txt            # 项目依赖
└── README.md                   # 项目文档
```

## 使用方法

### 1. 生成参考答案

首先，使用基线模型生成参考答案：

```bash
cd example_baseline
python generate_reference.py --input_path ./data/input.csv --output_path ./data/reference.csv
```

参数说明：
- `--input_path`: 输入数据路径
- `--output_path`: 参考答案输出路径
- `--model_name_or_path`: 模型名称或路径（默认：Qwen/Qwen3-4B）
- `--temperature`: 温度参数（默认：0.1，用于生成更确定性的答案）
- `--max_new_tokens`: 最大生成 token 数（默认：4096）

### 2. 运行主程序

使用压缩后的思维链生成答案：

```bash
python main.py --input_path ./data/input.csv --output_path ./data/output.csv --reference_path ./data/reference.csv
```

参数说明：
- `--input_path`: 输入数据路径
- `--output_path`: 输出数据路径
- `--reference_path`: 参考答案路径
- `--model_name_or_path`: 模型名称或路径（默认：Qwen/Qwen3-4B）
- `--system_prompt`: 系统提示词（默认使用压缩思维链的提示词）
- `--temperature`: 温度参数（默认：0.7）
- `--num_samples`: 每个问题生成的样本数量（默认：5）
- `--result_dir`: 结果保存目录（默认：./results）

### 3. 评估结果

程序会自动评估生成结果，包括：
- 准确率评估
- 思维链长度统计
- 可视化分析

评估结果将保存在 `result_dir` 目录下，包括：
- `evaluation_report.json`: 详细的评估报告
- `cot_length_distribution.png`: 思维链长度分布图
- `accuracy_pie.png`: 准确率饼图
- `length_vs_accuracy.png`: 思维链长度与准确率关系图

## 系统提示词

项目使用两种不同的系统提示词：

1. 生成参考答案时（详细思维链）：
```
Think step by step carefully. Show your detailed reasoning process. Return the final answer at the end of the response after a separator ####, wrapped in \boxed{}.
```

2. 压缩思维链时：
```
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.
```

## 注意事项

1. 确保有足够的 GPU 显存运行 Qwen3-4B 模型
2. 输入数据应为制表符分隔的 CSV 文件，包含问题文本
3. 建议使用 CUDA 环境以获得更好的性能
4. 如果遇到内存不足，可以调整 `max_new_tokens` 参数

## 许可证

[添加许可证信息]

## 联系方式

[添加联系方式] 