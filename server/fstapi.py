from typing import Union
import uvicorn
from fastapi import FastAPI
from hugf import hugf_Model
from img_to_txt_model import hugf_img_to_txt_Model
from video_to_txt_model import hugf_vid_to_txt_Model
from audio_to_txt_model import hugf_aud_to_txt_Model

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


#@app.get("/prompt/{prompt}")
@app.get("/prompt")
def call_model(prompt: Union[str, None] = 'Hello'):
    reply=hugf_Model(prompt)
    return {"reply": reply}


@app.get("/img_to_txt")
def call_model(img: Union[str, None] = ''):
    txt=hugf_img_to_txt_Model(img)
    return {"txt": txt}

@app.get("/vid_to_txt")
def call_model(vid: Union[str, None] = ''):
    txt=hugf_vid_to_txt_Model(vid)
    return {"txt": txt}

@app.get("/aud_to_txt")
def call_model(aud: Union[str, None] = ''):
    txt=hugf_aud_to_txt_Model(aud)
    return {"txt": txt}

if __name__ == "__main__":
    uvicorn.run(app, port=80)