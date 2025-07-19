from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with your model path
model_path = "/home/hzchen/jyh/llm_models/TIGER-Lab/general-verifier"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

# Example inputs
question = "Factor the following quadratic: $3 x^3+\\frac{69 x^2}{2}-36 x-810$"
ground_truth = "\\frac{3(2x-9)(x+6)(x+10)}{2}"
student_answer = "\\frac{3}{2}(2x-9)(x+6)(x+9)"

# Create prompt
prompt = (
    f"User: ### Question: {question}\n\n"
    f"### Ground Truth Answer: {ground_truth}\n\n"
    f"### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.0,
    do_sample=False
)

# Decode and print output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
