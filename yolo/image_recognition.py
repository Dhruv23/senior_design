import os 
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_PY = os.path.normpath(os.path.join(HERE, "../frame/frame.py"))

spec = importlib.util.spec_from_file_location("frame", FRAME_PY)
frame_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_mod)  # now frame_mod.capture_frame is available


import ultralytics

model = ultralytics.YOLO("yolov8n.pt") 

def main():
    # output directory 
    pics_dir = os.path.join(HERE, "pics")
    os.makedirs(pics_dir, exist_ok=True)


    img_path = os.path.join(pics_dir, "yolo_input.png")
    frame_mod.capture_frame(img_path)
    print(f"[INFO] Captured frame: {img_path}")


    model = ultralytics.YOLO("yolov8n.pt")
    results = model(img_path)   # predict on the image
    res = results[0]
    # print/display/save results

    res.show()              
    res.save()              
    print(res)


if __name__ == "__main__":
    main()
