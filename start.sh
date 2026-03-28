#!/bin/bash
source /home/ben/yolo_object/bin/activate

export FLASK_SECRET_KEY="change-me"
export GARDEFENDER_PASSWORD="change-me"
export GARDEFENDER_DETECTIONS="/home/ben/Desktop/Squirrel Cannon/detections"
export GARDEFENDER_MODEL="/home/ben/Desktop/Squirrel Cannon/yolo11n_ncnn_model"

cd /home/ben/Gardefender3000
python wsgi.py
