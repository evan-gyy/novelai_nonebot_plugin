import time
import os 
from IPython.display import clear_output, display
from contextlib import nullcontext
import paddle
from PIL import Image

_VAE_SIZE_THRESHOLD_ = 300000000       # vae should not be smaller than this
_MODEL_SIZE_THRESHOLD_ = 3000000000    # model should not be smaller than this

def empty_cache():
    """Empty CUDA cache. Essential in stable diffusion pipeline."""
    import gc
    gc.collect()
    paddle.device.cuda.empty_cache()

def check_is_model_complete(path = None, check_vae_size=_VAE_SIZE_THRESHOLD_):
    """Auto check whether a model is complete by checking the size of vae > check_vae_size.
    The vae of the model should be named by model_state.pdparams."""
    path = path or os.path.join('./',os.path.basename(model_get_default())).rstrip('.zip')
    return os.path.exists(os.path.join(path,'vae/model_state.pdparams')) and\
         os.path.getsize(os.path.join(path,'vae/model_state.pdparams')) > check_vae_size

def model_get_default(base_path = 'data'):
    """Return an absolute path of model zip file in the `base_path`."""
    available_models = []
    for folder in os.walk(base_path):
        for filename_ in folder[2]:
            filename = os.path.join(folder[0], filename_)
            if filename.endswith('.zip') and os.path.isfile(filename) and os.path.getsize(filename) > _MODEL_SIZE_THRESHOLD_:
                available_models.append((os.path.getsize(filename), filename, filename_))
    available_models.sort()
    # use the model with smallest size to save computation
    return available_models[0][1]

def model_vae_get_default(base_path = 'data'):
    """Return an absolute path of extra vae if there is any."""
    for folder in os.walk(base_path):
        for filename_ in folder[2]:
            filename = os.path.join(folder[0], filename_)
            if filename.endswith('vae.pdparams'):
                return filename
    return None

def model_unzip(abs_path = None, name = None, dest_path = './', verbose = True):
    """Unzip a model from `abs_path`, `name` is the model name after unzipping."""
    if abs_path is None:
        abs_path = model_get_default()
    if name is None:
        name = os.path.basename(abs_path)

    from zipfile import ZipFile
    dest = os.path.join(dest_path, name).rstrip('.zip')
    if not check_is_model_complete(dest):
        if os.path.exists(dest):
            # clear the incomplete zipfile
            if verbose: print('检测到模型文件破损, 正在删除......')
            import shutil
            shutil.rmtree(dest)
        
        if verbose: print('正在解压模型......')
        with ZipFile(abs_path, 'r') as f:
            f.extractall(dest_path)
    else:
        print('模型已存在')

def save_image_info(image, path = './outputs/'):
    """Save image to a path with arguments."""
    os.makedirs(path, exist_ok=True)
    cur_time = time.time()
    seed = image.argument['seed']
    filename = f'{cur_time}_SEED_{seed}'
    image.save(os.path.join(path, filename + '.png'), quality=100)
    with open(os.path.join(path, filename + '.txt'), 'w') as f:
        for key, value in image.argument.items():
            f.write(f'{key}: {value}\n')
    return filename + '.png'
    
def ReadImage(image, height = None, width = None):
    """Read an image and resize it to (height,width) if given.
    If (height,width) = (-1,-1), resize it so that 
    it has w,h being multiples of 64 and in medium size."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    max_edge = 576
    # clever auto inference of image size
    w, h = image.size
    if height == -1 or width == -1:
        if w > h:
            width = max_edge
            height = max(64, round(width / w * h / 64) * 64)
        else: # w < h
            height = max_edge
            width = max(64, round(height / h * w / 64) * 64)
        if width > 576 and height > 576:
            width = 576
            height = 576
    if (height is not None) and (width is not None) and (w != width or h != height):
        image = image.resize((width, height), Image.ANTIALIAS)
    return image

class StableDiffusionFriendlyPipeline():
    def __init__(self, model_name = None, superres_pipeline = None):
        self.pipe = None
        self.pipe_i2i = None
        self.model = model_name or os.path.basename(model_get_default()).rstrip('.zip')
        if not check_is_model_complete(self.model):
            assert (not os.path.exists(self.model)), self.model + '解压不完全! 请重启内核, 重新解压模型!'
            
        self.remote_vae = model_vae_get_default()
        self.vae = None if self.remote_vae is None else os.path.basename(self.remote_vae)

        self.superres_pipeline = superres_pipeline

    def from_pretrained(self, verbose = True, force = False):
        model = self.model
        vae = self.vae
        if (not force) and self.pipe is not None:
            return

        if verbose: print('!!!!!正在加载模型, 请耐心等待, 如果出现两行红字是正常的, 不要惊慌!!!!!')
        from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        # text to image
        self.pipe = StableDiffusionPipeline.from_pretrained(model)
        
        if vae is not None:
            print('正在换用 vae......')
            local_vae = os.path.join(os.path.join(self.model, 'vae'), self.vae)
            if (not os.path.exists(local_vae)) or os.path.getsize(local_vae) < _VAE_SIZE_THRESHOLD_:
                print('初次使用, 正在复制 vae...... (等 %s/vae/%s 文件约 319MB 即可)'%(self.model, self.vae))
                from shutil import copy
                copy(self.remote_vae, local_vae) # copy from remote, avoid download everytime

            self.pipe.vae.load_state_dict(paddle.load(local_vae)) # 换用更好的 vae (有效果!)

        # image to image
        pipe = self.pipe
        self.pipe_i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,
        unet=pipe.unet,scheduler=pipe.scheduler,safety_checker=pipe.safety_checker,feature_extractor=pipe.feature_extractor)

        # save space on GPU as we do not need them (safety check has been turned off)
        del self.pipe.safety_checker
        del self.pipe_i2i.safety_checker
        self.pipe.safety_checker = None
        self.pipe_i2i.safety_checker = None

        if verbose: print('成功加载完毕')

    def run(self, opt, task = 'txt2img'):
        self.from_pretrained()
        seed = None if opt.seed == -1 else opt.seed
        # precision_scope = paddle.amp.auto_cast if opt.precision=="autocast" else nullcontext
        # PRECISION = "fp16" if opt.precision=="autocast" else "fp32"

        task_func = None
        if task == 'txt2img':
            def task_func():
                return self.pipe(opt.prompt, seed=seed, width=opt.width, height=opt.height, guidence_scale=opt.guidence_scale, 
                                num_inference_steps=opt.num_inference_steps, negative_prompt=opt.negative_prompt).images[0]
        elif task == 'img2img':
            def task_func():
                return self.pipe_i2i(opt.prompt, seed=seed, 
                                init_image=ReadImage(opt.image_path, height=opt.height, width=opt.width), 
                                num_inference_steps=opt.num_inference_steps, 
                                strength=opt.strength, guidance_scale=opt.guidence_scale, negative_prompt=opt.negative_prompt)[0][0]
        
        with nullcontext():
            for i in range(opt.num_return_images):
                empty_cache()
                image = task_func()
                
                # super resolution
                if (self.superres_pipeline is not None):
                    argument = image.argument
                    argument['superres_model_name'] = opt.superres_model_name
                    
                    image = self.superres_pipeline.run(opt, image = image, end_to_end = False)
                    image.argument = argument

                save_image_info(image, opt.output_dir)
                if i % 5 == 0:
                    clear_output()

                display(image)
                print('Seed =', image.argument['seed'])

class SuperResolutionPipeline():
    def __init__(self):
        self.model = None
        self.model_name = ''
    
    def run(self, opt, 
                image = None, 
                task = 'superres', 
                end_to_end = True,
                force_empty_cache = True
            ):
        """
        end_to_end: return PIL image if False, display in the notebook and autosave otherwise
        empty_cache: force clear the GPU cache by deleting the model
        """
        if opt.superres_model_name is None or opt.superres_model_name in ('','无'):
            return image

        import numpy as np
        if image is None:
            image = ReadImage(opt.image_path, height=None, width=None) # avoid resizing
        image = np.array(image)
        image = image[:,:,[2,1,0]]  # RGB -> BGR

        empty_cache()
        if self.model_name != opt.superres_model_name:
            if self.model is not None:
                del self.model 

            import logging
            logging.disable(100)    
            # [ WARNING] - The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object
            import paddlehub as hub
            # print('正在加载超分模型! 如果出现两三行红字是正常的, 不要担心哦!')
            self.model = hub.Module(name = opt.superres_model_name)
            logging.disable(30)
            
        self.model_name = opt.superres_model_name

        # time.sleep(.1) # wait until the warning prints
        # print('正在超分......请耐心等待')
    
        try:
            image = self.model.reconstruct([image], use_gpu = (paddle.device.get_device() != 'cpu'))[0]['data']
        except:
            print('图片尺寸过大, 超分时超过显存限制')
            self.empty_cache(force_empty_cache)
            paddle.disable_static()
            return

        image = image[:,:,[2,1,0]] # BGR -> RGB
        image = Image.fromarray(image)
        
        self.empty_cache(force_empty_cache)
        paddle.disable_static()

        if end_to_end:
            cur_time = time.time()
            image.save(os.path.join(opt.output_dir,f'Highres_{cur_time}.png'),quality=100)
            clear_output()
            display(image)
            return
        return image
    
    def empty_cache(self, force = True):
        # NOTE: it seems that ordinary method cannot clear the cache
        # so we have to delete the model (?)
        if not force:
            return
        del self.model
        self.model = None
        self.model_name = ''


class StableDiffusionSafetyCheckerEmpty(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x