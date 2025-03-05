import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

model = AutoModel.from_pretrained('/home/modelscope/shan/MiniCPM-V-2_6/model_files', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/home/modelscope/shan/MiniCPM-V-2_6/model_files', trust_remote_code=True)

image1 = Image.open('image.jpg').convert('RGB')
image2 = Image.open('image2.png').convert('RGB')
question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'

msgs = [{'role': 'user', 'content': [image1, image2, question]}]

# First round chat
answer1 = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer1)

# Second round chat
# pass history context of multi-turn conversation

msgs.append({"role": "assistant", "content": [answer1]})
msgs.append({"role": "user", "content": ["Provide summary of differences of both images"]})


answer2 = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer2)



