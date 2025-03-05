import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download

model_dir = snapshot_download('LLM-Research/Llama-3.2-11B-Vision', ignore_file_pattern=['*.pth'])

model = MllamaForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir)

url = "https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision/resolve/master/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))

