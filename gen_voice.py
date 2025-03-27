import os
import aiofiles
from gradio_client import Client, file
from pathlib import Path

url = "http://localhost:9872/"


def tts(ref_wav_path,
              prompt_text, prompt_language, text,
              text_language, ):
    client = Client(url)
    result = client.predict(
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language=text_language,
        how_to_cut="凑四句一切",
        top_k=15,
        top_p=1,
        temperature=1,
        ref_free=False,
        speed=1,
        if_freeze=False,
        inp_refs=None,
        sample_steps=32,
        api_name="/get_tts_wav"
    )
    print(result)
    return result


def change_sovits_weights(sovits_path):
    client = Client(url)
    result = client.predict(
        sovits_path=sovits_path,
        prompt_language="中文",
        text_language="中文",
        api_name="/change_sovits_weights"
    )


def change_gpt_weights(gpt_path):
    client = Client(url)
    result = client.predict(
        gpt_path=gpt_path,
        api_name="/change_gpt_weights"
    )


def get_model_list():
    """获取可用模型列表并缓存"""
    client = Client(url)
    result = client.predict(api_name="/change_choices")
    return {
        "sovits": [choice[0] for choice in result[0]["choices"]],
        "gpt": [choice[0] for choice in result[1]["choices"]]
    }

