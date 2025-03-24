from pprint import pprint
from gradio_client import Client, file

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
    pprint(result)
    return result


def change_sovits_weights(sovits_path):
    client = Client(url)
    result = client.predict(
        sovits_path=sovits_path,
        prompt_language="中文",
        text_language="中文",
        api_name="/change_sovits_weights"
    )
    pprint(result)


def change_gpt_weights(gpt_path):
    client = Client(url)
    result = client.predict(
        gpt_path=gpt_path,
        api_name="/change_gpt_weights"
    )
    pprint(result)


def change_choices():
    client = Client(url)
    result = client.predict(
        api_name="/change_choices"
    )
    choices_vits = []
    choices_gpt = []
    for i in result[0]["choices"]:
        choices_vits.append(i[0])
    for i in result[1]["choices"]:
        choices_gpt.append(i[0])
    return choices_vits, choices_gpt
