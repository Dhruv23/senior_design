#!/usr/bin/env python3
import sys
import os
import importlib.util
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


#import open_camera from video.py 
HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_PY = os.path.normpath(os.path.join(HERE, "../video/video.py"))

spec = importlib.util.spec_from_file_location("video", VIDEO_PY)
video_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_mod)  # now video_mod.open_camera() is available


def main():
    cap = video_mod.open_camera() # loading from prewritten file

    # choose device: GPU if available, else CPU
    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"

    #load YOLO and move to device
    model = YOLO("yolov8n.pt")
    if use_cuda:
        model.to("cuda")
    use_half = bool(use_cuda)  # FP16 only on CUDA

    print(f"press 'q' to quit  |  Device: {'CUDA:0' if use_cuda else 'CPU'}  |  FP16: {use_half}")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("failed to grab frame")
            break

        # Inference on the selected device (FP16 when on GPU)
        results = model(
            frame,
            verbose=False,
            device=device,
            half=use_half
        )
        res = results[0]

        # Draw only boxes with conf >= 0.8
        annotator = Annotator(frame)
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf >= 0.8:
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                xyxy = box.xyxy[0]
                annotator.box_label(xyxy, label, color=(0, 255, 0))

        cv2.imshow("YOLOv8 Live (>=0.8 conf)", annotator.result())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
