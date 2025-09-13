#!/usr/bin/env python3
import cv2
import sys

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # /dev/video0
    if not cap.isOpened():
        print("Could not open /dev/video0")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


def main():
    cap = open_camera()

    print("press q to quit the video window.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("failed to grab frame")
            break

        cv2.imshow("Live Feed", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
