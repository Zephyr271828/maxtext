import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 配置参数
model_name = "/tmp/llama3_4b_width_hf"  # 需为 HF 上的路径或本地路径
prompt = "I love to"
max_prefill_len = 4
max_target_len = 5  # inclusive
device = xm.xla_device()

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cpu')
model.to(device)
model.eval()

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]  # shape [1, T]
attention_mask = inputs["attention_mask"]

# 手动截断前 prefill 长度
input_ids = input_ids[:, :max_prefill_len]
attention_mask = attention_mask[:, :max_prefill_len]

# 🔹 Step 1: Prefill logits
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits = output.logits  # [1, T, vocab_size]
    past_key_values = output.past_key_values

print("Prefill logits shape:", logits.shape)  # [1, T, V]

# 取最后一个 token 的 logits，采样第一个 token
last_logits = logits[:, -1, :]  # [1, V]
next_token = torch.argmax(last_logits, dim=-1, keepdim=True)  # greedy
generated = [next_token]

# 🔹 Step 2: Autoregressive decoding
for _ in range(max_prefill_len, max_target_len):
    input_ids = next_token  # shape [1, 1]
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=None,  # decode step 不需要 mask
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = output.logits  # [1, 1, V]
        print('logits:', logits)
        past_key_values = output.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)

# 拼接结果
generated_tokens = torch.cat(generated, dim=-1)
full_tokens = torch.cat([inputs["input_ids"], generated_tokens], dim=-1)
decoded = tokenizer.decode(full_tokens[0], skip_special_tokens=True)
print("Decoded text:", decoded)