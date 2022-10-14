#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
PROJECT_PATH = "F:/Project/paddle"  # 修改成你的项目路径
sys.path.append(PROJECT_PATH)
from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import paddle
from utils import save_image_info
import os

class NovelAI:
    def __init__(self):
        self.pipe = None
        self.pipe_i2i = None


    def load_pipe(self):
        pipe = StableDiffusionPipeline.from_pretrained(PROJECT_PATH + "/NovelAI_latest_ab21ba3c_paddle")
        # pipe = StableDiffusionPipeline.from_pretrained("./model_pruned_paddle")

        vae_path = PROJECT_PATH + '/NovelAI_latest_ab21ba3c_paddle/vae/animevae.pdparams'
        vae_path = vae_path if os.path.exists(vae_path) else PROJECT_PATH + '/data/data171442/animevae.pdparams' # 加载 vae
        pipe.vae.load_state_dict(paddle.load(vae_path)) # 换用更好的 vae (有效果!)

        # 图生图
        pipe_i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,
        unet=pipe.unet,scheduler=pipe.scheduler,safety_checker=pipe.safety_checker,feature_extractor=pipe.feature_extractor)
        print('加载完毕')

        self.pipe = pipe
        self.pipe_i2i = pipe_i2i


    def txt2img(self,
                prompt="miku, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress",
                negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                height=512,
                width=512,
                seed=None,
                steps=50,
                cfg=7
    ):
        image = self.pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=cfg, seed=seed,
                     negative_prompt=negative_prompt).images[0]
        print('Seed =', image.argument['seed'])
        path = save_image_info(image, path=PROJECT_PATH + '/outputs/')  # 保存图片到指定位置
        return image, path


    def img2img(self,
                path = "image_Kurisu.png",
                steps = 50,
                seed = 20221013,
                strength = 0.8,
                cfg = 7.5,
                prompt = "Kurisu Makise",
                negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    ):
        from utils import ReadImage
        init_image = ReadImage(path, height=-1, width=-1)  # -1 为自动判断图片大小
        image2 = \
        self.pipe_i2i(prompt, init_image=init_image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,
                 seed=seed, negative_prompt=negative_prompt)[0][0]
        print('Seed =', image2.argument['seed'])
        path = save_image_info(image2, path=PROJECT_PATH + '/outputs/')  # 保存图片到指定位置
        return image2, path


if __name__ == '__main__':
    nv = NovelAI()
    nv.load_pipe()
    nv.txt2img()