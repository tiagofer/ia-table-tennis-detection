#!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="MzHZkio68xGH6Se3iFIn")
project = rf.workspace("karl-paul-parmakson-dxvbj").project("tt-table-segmentation")
dataset = project.version(1).download("yolov8")


