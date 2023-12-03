import os
import cv2

import argparse
import numpy as np
import pandas as pd
from collections import deque
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str)
parser.add_argument('--save_dir', type=str, default='outputs/')
args = parser.parse_args()

video_file = args.video_file
save_dir = args.save_dir
video_name = video_file.split('/')[-1][:-4]
output_video_file = f'{save_dir}/{video_name}_chart.mp4'


# Configurações do gráfico
graph_color = 'green'  # Cor verde para o gráfico
graph_thickness = 2  # Espessura da linha do gráfico
graph_scale = 0.2  # Escala do gráfico em relação à imagem original

# Cap configuration
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, fps, (w, h))

frame_i = 0
df = pd.read_csv('outputs/keypoints.csv')[['player','frame','right_wrist_x','right_wrist_y','right_elbow_x','right_elbow_y','right_shoulder_x','right_shoulder_y']]
df['player'] = np.where(df.right_shoulder_x >= 0.5, 1, 0)
df_p1 = df[df['player'] == 0].copy().round(2)
df_p2 = df[df['player'] == 1].copy().round(2)

# Configurar as coordenadas onde a imagem será inserida
posicao_x = 100  # coordenada X da imagem na tela
posicao_y = 100  # coordenada Y da imagem na tela

def draw_keypoints(lines, frame, x_position, y_position,color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 1
    font_color = color
    # Posição inicial do texto
    y_position = 100

    # Adiciona cada linha ao frame
    for line in lines:
        cv2.putText(frame, line, (x_position, y_position), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        y_position += 50  # Ajuste conforme necessário para o espaçamento entre linhas


while success:
    success, frame = cap.read()
    if not success:
        break
    try:
        last_row = df_p1[df_p1['frame'] == frame_i].iloc[-1]
      # Plota o gráfico no frame
        if not last_row.empty:
            
            lines = [
                f"Frame: {frame_i} Player 1",\
                f"Wrist X: {last_row['right_wrist_x']}, Wrist Y: {last_row['right_wrist_y']}",\
                f"Elbow X: {last_row['right_elbow_x']}, Elbow Y: {last_row['right_elbow_y']}",\
                f"Shoulder X: {last_row['right_shoulder_x']}, Shoulder Y: {last_row['right_shoulder_y']}"]
            draw_keypoints(lines,frame,30,400,(0, 255, 0))
    except:
        lines = [
                f"Frame: {frame_i} Player 2",\
                f"Wrist X: 0, Wrist Y: 0",\
                f"Elbow X: 0, Elbow Y: 0",\
                f"Shoulder X: 0, Shoulder Y: 0"]
        draw_keypoints(lines,frame,30,400,(0, 255, 0))    
    
    try:  
        last_row_p2 = df_p2[df_p2['frame'] == frame_i].iloc[-1]   
        if not last_row_p2.empty:   
            lines_p2 = [
                f"Frame: {frame_i} Player 2",\
                f"Wrist X: {last_row_p2['right_wrist_x']}, Wrist Y: {last_row_p2['right_wrist_y']}",\
                f"Elbow X: {last_row_p2['right_elbow_x']}, Elbow Y: {last_row_p2['right_elbow_y']}",\
                f"Shoulder X: {last_row_p2['right_shoulder_x']}, Shoulder Y: {last_row_p2['right_shoulder_y']}"]
            draw_keypoints(lines_p2,frame,1400,400,(0, 0, 255))     
    except:
        lines = [
                f"Frame: {frame_i} Player 2",\
                f"Wrist X: 0, Wrist Y: 0",\
                f"Elbow X: 0, Elbow Y: 0",\
                f"Shoulder X: 0, Shoulder Y: 0"]
        draw_keypoints(lines,frame,1400,400,(0, 0, 255))  
    out.write(frame)

    #frame_pol =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #out.write(graph_image)
    frame_i += 1



out.release()
cap.release()
print('Done')