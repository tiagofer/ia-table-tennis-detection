import os
import cv2

import argparse
import numpy as np
import pandas as pd
from collections import deque
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str)
parser.add_argument('--box_coordinates', type=str)
parser.add_argument('--save_dir', type=str, default='outputs/')
args = parser.parse_args()

video_file = args.video_file
save_dir = args.save_dir
video_name = video_file.split('/')[-1][:-4]
output_video_file = f'{save_dir}/{video_name}_table.mp4'


box_coordinates = np.load(args.box_coordinates)
print(box_coordinates)
# Cap configuration
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, fps, (w, h))

frame_i = 0
while success:
    success, frame = cap.read()
    if not success:
        break

    # Convert to PIL image for drawing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    polygon_points = [(box_coordinates[i], box_coordinates[i + 1]) for i in range(0, len(box_coordinates), 2)]

    draw.polygon(polygon_points,outline='yellow', width=5)
    frame_pol =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    out.write(frame_pol)



out.release()
cap.release()
print('Done')