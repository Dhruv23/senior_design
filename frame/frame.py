#!/usr/bin/env python3
import sys
import cv2

import os
def capture_frame(output_file: str):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # /dev/video0
    if not cap.isOpened():
        print("could not open /dev/video0")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("failed to capture frame")
        sys.exit(1)

    cv2.imwrite(output_file, frame) # imwrite method can handle most image formats.
    print(f"Saved frame to {output_file}")

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "frame.png"
    if "." not in output.lower():
        output += ".png"

    os.makedirs("pics", exist_ok=True)
    output = "pics/" + output
    
    capture_frame(output)
