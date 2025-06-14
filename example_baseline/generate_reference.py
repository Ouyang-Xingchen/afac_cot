"""
生成参考答案脚本

使用基线模型（Qwen3-4B）生成参考答案，用于后续评估。
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from model_inference import ModelInference
from data_loader import DataLoader

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成参考答案")
    parser.add_argument("--input_path", type=str, default="./data/input.csv", help="输入数据路径")
    parser.add_argument("--output_path", type=str, default="./data/reference.csv", help="参考答案输出路径")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B", help="模型名称或路径")
    parser.add_argument("--system_prompt", type=str, 
                       default="Think step by step carefully. Show your detailed reasoning process. Return the final answer at the end of the response after a separator ####, wrapped in \\boxed{}.",
                       help="系统提示词")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.1, help="温度参数（使用较低的温度以获得更确定性的答案）")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数")
    parser.add_argument("--num_samples", type=int, default=1, help="每个问题生成的样本数量（参考答案通常只需要1个）")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 加载数据
    print("=== 加载数据 ===")
    loader = DataLoader(args.input_path)
    questions = loader.get_questions()
    print(f"共加载 {len(questions)} 个问题")
    
    # 准备prompts
    prompts = loader.prepare_batch_prompts(args.system_prompt)
    
    # 加载模型
    print("\n=== 加载模型 ===")
    inference = ModelInference(args.model_name_or_path)
    inference.load_model()
    
    # 生成答案
    print("\n=== 生成参考答案 ===")
    results = inference.batch_inference(
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples
    )
    
    # 提取答案
    print("\n=== 处理结果 ===")
    reference_data = []
    for i, (question_id, result) in enumerate(zip(questions['id'], results)):
        # 获取第一个样本的答案（因为num_samples=1）
        output = result[0]
        
        # 提取\boxed{}中的答案
        import re
        pattern = r'\\boxed{([^}]*)}'
        matches = list(re.finditer(pattern, output))
        if matches:
            answer = matches[-1].group(1).strip()
        else:
            print(f"警告：问题 {question_id} 未找到答案")
            answer = ""
        
        reference_data.append({
            'id': question_id,
            'answer': answer,
            'full_output': output  # 保存完整输出以供参考
        })
    
    # 保存结果
    reference_df = pd.DataFrame(reference_data)
    reference_df.to_csv(args.output_path, index=False)
    print(f"\n参考答案已保存至: {args.output_path}")
    
    # 打印统计信息
    total_questions = len(reference_data)
    answered_questions = sum(1 for item in reference_data if item['answer'])
    print(f"\n=== 统计信息 ===")
    print(f"总问题数: {total_questions}")
    print(f"已回答问题数: {answered_questions}")
    print(f"答案覆盖率: {answered_questions/total_questions:.2%}")

if __name__ == "__main__":
    main() 