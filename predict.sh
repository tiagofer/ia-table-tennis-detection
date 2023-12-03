#!/bin/sh
#python3 tracknetv2/3_in_3_out/predict4.py --video_name=input/teste_tiago.mp4 --load_weights=tracknetv2/3_in_3_out/model906_30
python3 TrackNetV2/predict.py --video_file input/output1.mp4 --model_file TrackNetV2/models/model_best.pt --save_dir outputs/
python3 predict_table.py input/output1.mp4 ..
python3 table/show_table.py --video_file outputs/output1_pred.mp4 --box_coordinates outputs/box_coordinates.npy --save_dir outputs/
python3 predict_keypoints.py outputs/output1_pred_table.mp4 outputs/keypoints.csv
python3 utils.py --video_file outputs/output1_pred_table.mp4 --save_dir outputs/
#mv /usr/src/ultralytics/runs/pose/predict/teste_tiago_pred_table_chart.avi outputs/teste_tiago_pred_table_chart_keys.avi