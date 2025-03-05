import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord



device_ids = [0,1,2,3]
torch.cuda.set_device(device_ids[0])


model = AutoModel.from_pretrained(
    '/home/modelscope/shan/MiniCPM-V-2_6/model/', 
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16
) # sdpa or flash_attention_2, no eager

model = model.eval()
model = torch.nn.DataParallel(model, device_ids=device_ids)


if torch.cuda.is_available():
    model = model.cuda()

tokenizer = AutoTokenizer.from_pretrained(
    '/home/modelscope/shan/MiniCPM-V-2_6/model/', 
    trust_remote_code=True
)


MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path ="video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]}, 
]

# Set decode params for video
params={}
params["use_image_id"] = False
params["max_slice_nums"] = 1 # use 1 if cuda OOM and video resolution >  448*448

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)
print(answer)
