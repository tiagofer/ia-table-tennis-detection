import os
import queue
import cv2
import numpy as np
from PIL import ImageDraw
from PIL import Image
import csv
import sys
from ultralytics import YOLO
import pandas as pd
import torch

torch.cuda.set_device(0)

try:
	input_video_path = sys.argv[1]
	input_csv_path = sys.argv[2]
	#output_video_path = sys.argv[3]
	if (not input_video_path) or (not input_csv_path):
		raise ''
except:
	print('usage: python3 main.py <input_video_path> <input_csv_path>')
	exit(1)
 
#get video fps&video size
currentFrame= 0
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))

video.set(1,currentFrame); 
ret, img1 = video.read()
#write image to video
#output_video.write(img1)
currentFrame +=1
#input must be float type
#img1 = img1.astype(np.float32)

# Load a model
model = YOLO('table/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(img1,imgsz=640, conf=0.9,device=0)  # return a list of Results objects

masks = results[0].masks
mask1 = masks[0]

mask = mask1.cpu().data[0].numpy()
polygon = mask1.xy[0]


#pd.DataFrame(polygon).to_csv('outputs/table.csv')
# for r in results:
#     mask_all = r.masks
#     mask = mask_all.data.numpy()
#     polygon = mask_all.xy

mask_img = Image.fromarray(mask,"I")
mask_img.save('outputs/mask.png')


epsilon = 0.01*cv2.arcLength(polygon,True)
approx = cv2.approxPolyDP(polygon,epsilon,True)

"""
0,0----------------width,0
0,height-----------width,height
[[[       1374         591]] top-right

 [[        954         576]] bottom-right

 [[        372         702]] top-left

 [[        822         789]]] 
"""

def scale_contour(cnt, scale):
    cnt = np.array(cnt)
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

approx = scale_contour(approx,1.05)

# You may need to convert the color.
font_face = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
color = (255, 255, 255)
thickness = cv2.FILLED
margin = 50


print(approx)
p1 = [int(approx[0][0][0]),int(approx[0][0][1])]
p2 = [int(approx[1][0][0]),int(approx[1][0][1])]
p3 = [int(approx[2][0][0]),int(approx[2][0][1])]
p4 = [int(approx[3][0][0]),int(approx[3][0][1])]

p1_p4 = p1[1] - p4[1]
p2_p3 = p2[1] - p3[1]
diff = abs(p1_p4 - p2_p3)
print(p1_p4, p2_p3,diff)

if abs(p1_p4) > abs(p2_p3):
    p2[1] = int(p2[1] + diff/2)
    p3[1] = int(p3[1] - diff/2)

scale_factor = 0
box_coordinates = [p1[0]+scale_factor,p1[1],\
            p2[0]+scale_factor,p2[1],\
            p3[0]-scale_factor,p3[1],\
            p4[0]-scale_factor,p4[1]]

for p in approx:
    text = f'P[{p[0][0],p[0][1]}]'
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    #cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img1, text, (int(p[0][0]),int(p[0][1])), font_face, scale, color, 1, cv2.LINE_AA)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img1)
draw = ImageDraw.Draw(im_pil)
draw.polygon(box_coordinates,outline=(0,255,0), width=5)
im_pil.save('outputs/polygon.png')
np.save('outputs/box_coordinates',np.array(box_coordinates))
# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     #im.show()  # show image
#     im.save('outputs/table.jpg')  # save image