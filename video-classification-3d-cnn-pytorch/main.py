import os
import sys
import json
import subprocess
import shutil
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model, map_location=torch.device('cpu'))  # 添加 map_location 以支持 CPU
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row.strip())  # 修复潜在换行符问题

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row.strip())

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    # 删除 tmp 文件夹（替换 rm -rf tmp）
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')

    outputs = []
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)

            # 创建 tmp 文件夹（替换 mkdir tmp）
            os.makedirs('tmp', exist_ok=True)

            subprocess.call(f'ffmpeg -i {video_path} tmp/image_%05d.jpg -loglevel {ffmpeg_loglevel}', shell=True)

            result = classify_video('tmp', input_file, class_names, model, opt)
            outputs.append(result)

            # 删除 tmp 文件夹（替换 rm -rf tmp）
            shutil.rmtree('tmp')
        else:
            print('{} does not exist'.format(input_file))

    # 最后删除 tmp 文件夹（替换 rm -rf tmp）
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)