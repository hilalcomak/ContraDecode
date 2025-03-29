import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, skip_special_tokens=False)
txt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',
                                             quantization_config = BitsAndBytesConfig(load_in_4bit=True),
                                             torch_dtype=torch.bfloat16)

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    tokenizer=tokenizer,
    device_map="auto",
)


outputs = pipe(
    txt,
    max_new_tokens=256,
)

print(outputs)
