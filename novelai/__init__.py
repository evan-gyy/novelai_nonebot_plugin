# -*- coding: utf-8 -*-
from .novelai import NovelAI
import time
import json
import io
import requests
from PIL import Image
import nonebot
from nonebot import Driver
from nonebot.plugin import on_command
from nonebot.typing import T_State
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, Message, MessageSegment, Event
from nonebot.log import logger
from nonebot.params import CommandArg, Arg, ArgStr, Depends

__zx_plugin_name__ = "NovelAI绘图"
__plugin_usage__ = """
NovelAI绘图

指令：
nvt2i: 帮助
nvt2i [prompt] || ?[negative prompt]: 文生图
nvi2i ?[prompt] [图片]: 图生图 

示例：
nvt2i miku, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress
""".strip()
__plugin_des__ = "基于NovelAI的AI绘图"
__plugin_cmd__ = ["nvt2i/nvi2i"]
__plugin_type__ = ("一些工具",)
__plugin_version__ = 0.1
__plugin_author__ = "evan-gyy"
__plugin_settings__ = {
    "level": 5,
    "default_status": True,
    "limit_superuser": False,
    "cmd": __plugin_cmd__,
}

driver: Driver = nonebot.get_driver()
nv = NovelAI()

@driver.on_startup
async def init_novel_ai():
    global nv
    try:
        nv.load_pipe()
        return True
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")
        return False

novel_t2i = on_command("nvt2i", block=True, priority=5)
novel_i2i = on_command("nvi2i", block=True, priority=5)


@novel_t2i.handle()
async def _(bot: Bot, event: Event, args: Message = CommandArg()):
    text = args.extract_plain_text().strip()
    if not text:
        await novel_t2i.finish(__plugin_usage__)
    else:
        if "||" in text and text.count("||") == 1:
            prompt, negative_prompt = text.split("||")
        else:
            prompt = text
            negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        await novel_t2i.send("少女绘画中...")
        try:
            start = time.time()
            image, path = nv.txt2img(prompt=prompt, negative_prompt=negative_prompt)
            img_bytes = img2bytes(image)
            end = time.time()
            time_delta = str(round(end - start, 2)) + "s"
            msg = Message(f"prompt: {text}\nfinish: {time_delta}")
            msg += MessageSegment.image(img_bytes)
            await novel_t2i.send(msg)
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            await novel_t2i.send("你画给我看！")


@novel_i2i.handle()
async def _(bot: Bot, event: MessageEvent, state: T_State,args: Message = CommandArg()):
    prompt = args.extract_plain_text().strip()
    if prompt:
        state['prompt'] = prompt
    else:
        state['prompt'] = "1girl, detailed, masterpiece, beautifully detailed face"
    if get_message_img(event.json()):
        state["img"] = event.message


def parse_image(key: str):
    async def _key_parser(
        state: T_State, img: Message = Arg(key)
    ):
        if not get_message_img(img):
            await novel_i2i.reject_arg(key, "请发送要生成的图片！")
        state[key] = img
    return _key_parser


@novel_i2i.got("img", prompt="图来", parameterless=[Depends(parse_image("img"))])
async def _(bot: Bot,
            event: MessageEvent,
            state: T_State,
            text: str = ArgStr("prompt"),
            img: Message = Arg("img")
):
    url = get_message_img(img)[0]
    temp = io.BytesIO(requests.get(url).content)
    img = Image.open(temp)
    logger.info(img.size)
    if "||" in text and text.count("||") == 1:
        prompt, negative_prompt = text.split("||")
    else:
        prompt = text
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    await novel_i2i.send("少女绘画中...")
    try:
        start = time.time()
        image, path = nv.img2img(path=img, prompt=prompt, negative_prompt=negative_prompt)
        img_bytes = img2bytes(image)
        end = time.time()
        time_delta = str(round(end - start, 2)) + "s"
        msg = Message(f"prompt: {prompt}\nfinish: {time_delta}")
        msg += MessageSegment.image(img_bytes)
        await novel_i2i.send(msg)
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")
        await novel_i2i.send("你画给我看！")


def get_message_img(data):
    img_list = []
    if isinstance(data, str):
        data = json.loads(data)
        for msg in data["message"]:
            if msg["type"] == "image":
                img_list.append(msg["data"]["url"])
    else:
        for seg in data["image"]:
            img_list.append(seg.data["url"])
    return img_list


def img2bytes(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="png")
    img_bytes = img_bytes.getvalue()
    return img_bytes
