import lmdeploy
from lmdeploy.vl import load_image


pipe = lmdeploy.pipeline("pretrained/InternVL2-1B")
#image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(["Hi, pls intro yourself","Shanghai is"])
#response = pipe(('describe this image', image))
#print(response.text)
print(response)



# Serve LLM on Multiple GPUs Locally with LMDeploy
#lmdeploy serve api_server OpenGVLab/InternVL2-8B --server-port 23333 
#Run above command on command line.









