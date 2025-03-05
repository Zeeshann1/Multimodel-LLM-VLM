from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from modelscope import snapshot_download
model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct", local_dir = "/home/modelscope/shan/Qwen2-VL-7B-Instruct/model-files")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_dir)


# Inference

from PIL import Image
image = Image.open("/home/modelscope/shan/Qwen2-VL-7B-Instruct/image.png").convert("RGB")
image2 = Image.open("/home/modelscope/shan/Qwen2-VL-7B-Instruct/image2.jpg").convert("RGB")


# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "image", "image": image2},
            {"type": "text", "text": "describe both images and its differences"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)







