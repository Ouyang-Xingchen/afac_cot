import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # 假设你使用 transformers 库

class ModelInference:
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-4B"): # 假设Qwen-7B-Chat是你要加载的模型
        """
        初始化模型推理类。
        
        参数:
            model_name_or_path: 模型名称或路径，默认为"Qwen/Qwen-7B-Chat"
        """
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detected device: {self.device}")

        # 在这里或一个单独的方法中加载模型和tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print(f"Loading model and tokenizer from {self.model_name_or_path}...")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        
        # 加载模型
        # 对于大型模型，推荐使用 device_map="auto" 让 transformers 自动处理设备分配
        # 否则，你需要手动将模型移动到 self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16, # 或者 torch.float16 以节省显存，Qwen推荐bfloat16
            device_map="auto", # 这会智能地将模型层分配到可用的GPU上，如果GPU不够，会自动使用CPU
            trust_remote_code=True
        )

        # 如果没有使用 device_map="auto"，则需要手动移动模型到指定设备
        # 例如：self.model.to(self.device)
        # 但是，对于大型模型，device_map="auto" 是更推荐和智能的方式。

        print(f"Model loaded. Model's current device: {next(self.model.parameters()).device}")


    def generate_response(self, prompt: str, max_new_tokens: int = 100):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer are not loaded. Call _load_model_and_tokenizer() first.")

        # 将输入文本进行 tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # !!! 关键步骤2: 将模型的输入数据移动到同一个设备上 !!!
        # 遍历 inputs 字典中的所有 tensor，并移动到 self.device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print(f"Input tensors moved to device: {next(iter(inputs.values())).device}")

        # 进行推理
        with torch.no_grad(): # 推理时关闭梯度计算，节省显存
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# 示例使用
if __name__ == "__main__":
    # 请确保你的环境中安装了 transformers 和 torch
    # pip install transformers torch
    # 对于Qwen模型，可能还需要安装 accelerate, einops, tiktoken 等
    # pip install accelerate einops tiktoken

    # 实例化模型推理类
    # 如果是本地模型路径，model_name_or_path 就是你下载的Qwen模型文件夹路径
    # 例如：model_path = "/path/to/your/Qwen/Qwen-7B-Chat"
    inference_engine = ModelInference(model_name_or_path="Qwen/Qwen3-4B") 
    # 如果是Qwen/Qwen3-4B，确保hf上可访问或者本地已下载

    # 进行推理
    user_prompt = "请用中文写一个关于人工智能的短故事。"
    generated_text = inference_engine.generate_response(user_prompt)
    print("\n--- Generated Response ---")
    print(generated_text)

    # 验证模型是否真的在GPU上运行：
    # 如果上面 _load_model_and_tokenizer 的打印显示设备是 cuda，
    # 且 generate_response 中输入的 tensors 也显示在 cuda，
    # 那么模型就在GPU上运行了。
