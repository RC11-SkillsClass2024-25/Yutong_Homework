import os
import sys
import json
import subprocess
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_fps(video_file_path, frames_directory_path):
    p = subprocess.Popen('ffprobe -i "{}"'.format(video_file_path),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, res = p.communicate()
    res = res.decode('utf-8')

    duration_index = res.find('Duration:')
    if duration_index == -1:
        raise ValueError("Could not retrieve video duration from ffprobe output")

    duration_str = res[(duration_index + 10):(duration_index + 21)]
    hour = float(duration_str[0:2])
    minute = float(duration_str[3:5])
    sec = float(duration_str[6:10])
    total_sec = hour * 3600 + minute * 60 + sec

    n_frames = len(os.listdir(frames_directory_path))
    fps = round(n_frames / total_sec, 2)
    return fps

if __name__ == '__main__':
    result_json_path = sys.argv[1]
    video_root_path = sys.argv[2]
    dst_directory_path = sys.argv[3]
    if not os.path.exists(dst_directory_path):
        os.makedirs(dst_directory_path)
    
    class_name_path = sys.argv[4]
    temporal_unit = int(sys.argv[5])

    with open(result_json_path, 'r') as f:
        results = json.load(f)

    with open(class_name_path, 'r') as f:
        class_names = [row.strip() for row in f]

    for index in range(len(results)):
        video_path = os.path.join(video_root_path, results[index]['video'])
        print(video_path)

        clips = results[index]['clips']
        unit_classes = []
        unit_segments = []
        if temporal_unit == 0:
            unit = len(clips)
        else:
            unit = temporal_unit
        for i in range(0, len(clips), unit):
            n_elements = min(unit, len(clips) - i)
            scores = np.array(clips[i]['scores'])
            for j in range(i, i + n_elements):
                scores += np.array(clips[j]['scores'])
            scores /= n_elements
            unit_classes.append(class_names[np.argmax(scores)])
            unit_segments.append([clips[i]['segment'][0],
                                  clips[i + n_elements - 1]['segment'][1]])

        if os.path.exists('tmp'):
            shutil.rmtree('tmp', ignore_errors=True)
        os.makedirs('tmp')

        subprocess.call('ffmpeg -i "{}" tmp/image_%05d.jpg'.format(video_path), shell=True)

        fps = get_fps(video_path, 'tmp')

        for i in range(len(unit_classes)):
            for j in range(unit_segments[i][0], unit_segments[i][1] + 1):
                image_path = 'tmp/image_{:05}.jpg'.format(j)
                if not os.path.exists(image_path):
                    continue
                image = Image.open(image_path).convert('RGB')
                min_length = min(image.size)
                font_size = int(min_length * 0.05)
                font = ImageFont.truetype("arial.ttf", font_size)
                d = ImageDraw.Draw(image)
                bbox = d.textbbox((0, 0), unit_classes[i], font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x, y = int(font_size * 0.5), int(font_size * 0.25)
                rect_position = (x, y, x + text_width + x * 2, y + text_height + y * 2)
                d.rectangle(rect_position, fill=(30, 30, 30))
                d.text((x + x, y + y), unit_classes[i], font=font, fill=(235, 235, 235))
                image.save('tmp/image_{:05}_pred.jpg'.format(j))

        dst_file_path = os.path.join(dst_directory_path, os.path.basename(video_path))
        subprocess.call('ffmpeg -y -r {} -i tmp/image_%05d_pred.jpg -b:v 1000k "{}"'.format(fps, dst_file_path), shell=True)

        if os.path.exists('tmp'):
            shutil.rmtree('tmp', ignore_errors=True)
