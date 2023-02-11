# NovelAI绘图插件

基于paddle的novelai绘图插件，适配nonebot

> 注：该插件需要部署本地novelai环境，适合本地运行，可能不是最优的novelai插件部署方法，且搭建过程较为繁琐；如果不想耗费大量时间搭建本地环境，可以参考其他优秀开发者的基于novel-api的nonebot插件

## 准备工作

1. 确保有一个可运行的bot，bot搭建参考[NoneBot文档](https://v2.nonebot.dev/docs/)
2. 确保你的运行环境有一块显存较大的n卡，当然显存不够也可以通过调参解决

## 环境搭建

1. 下载paddle版本的novelai包：[四步教你用NovelAI生成二次元小姐姐](https://aistudio.baidu.com/aistudio/projectdetail/4710661)
2. 参考其中的`main.ipynb`进行文件解压、python运行环境的初步搭建
3. 安装`paddle`库，根据[PaddlePaddle安装指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)安装GPU版本，并确保和你的GUDA、cuDNN版本对应，安装完后可以通过`paddle.utils.run_check()`检查环境是否安装正确
4. 在`novelai/novelai.py`中将项目路径`PROJECT_PATH`修改为你的项目下载路径

## 使用

将`novelai`文件夹放在`bot/plugins`目录下，运行bot即可

```python
指令：
    nvt2i: 帮助
    nvt2i [prompt] || ?[negative prompt]: 文生图
    nvi2i ?[prompt] [图片]: 图生图 

示例：
    nvt2i miku, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress
```

## 输出

输出的图片一般会保存在`PROJECT_PATH/outputs`下，包含图片及其生成时使用的参数

## 一键启动

可以使用`启动bot+go.bat`一键启动bot和gocqhttp环境，运行前需要将`BOT_PATH`和`GO_PATH`修改为你的bot、gocqhttp目录
