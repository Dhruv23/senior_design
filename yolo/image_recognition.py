import os 
import sys
import importlib.util
import cv2
from ultralytics.utils.plotting import Annotator


HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_PY = os.path.normpath(os.path.join(HERE, "../frame/frame.py"))

spec = importlib.util.spec_from_file_location("frame", FRAME_PY)
frame_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_mod)  # now frame_mod.capture_frame is available


import ultralytics

model = ultralytics.YOLO("yolov8n.pt") 


def show_high_conf(res, model, conf_thresh=0.8):
    im = res.orig_img.copy()  # start with the original image
    annotator = Annotator(im)

    for box in res.boxes:
        conf = float(box.conf[0])
        if conf >= conf_thresh:
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            xyxy = box.xyxy[0]
            annotator.box_label(xyxy, label, color=(255, 0, 0))  # red boxes

    # show window
    cv2.imshow("Filtered Detections", annotator.result())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # also save filtered result
    import os
    out_dir = os.path.join("runs", "detect", "filtered")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "filtered.png")
    cv2.imwrite(out_path, annotator.result())
    print(f"Saved filtered result to {out_path}")

def main():
    # output directory 
    pics_dir = os.path.join(HERE, "pics")
    os.makedirs(pics_dir, exist_ok=True)


    img_path = os.path.join(pics_dir, "yolo_input.png")
    frame_mod.capture_frame(img_path)
    print(f"Captured frame: {img_path}")


    model = ultralytics.YOLO("yolov8n.pt")
    results = model(img_path)   # predict on the image
    res = results[0]

    # console summary of high-confidence boxes
    for box in res.boxes:
        if float(box.conf[0]) >= 0.8:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            print(f"{label} {float(box.conf[0]):.2f} at {xyxy}")

    # show only filtered boxes
    show_high_conf(res, model, conf_thresh=0.8)



if __name__ == "__main__":
    main()
