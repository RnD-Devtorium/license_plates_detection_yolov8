import cv2 as cv
from glob import glob
from ultralytics import YOLO


videos = glob('inputs/*.mp4')
print(videos)

model_pretrained = YOLO('yolov8n.pt')

video = cv.VideoCapture(videos[3])

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./outputs/video_result.avi', fourcc, 20.0, size)

ret = True

while ret:
    ret, frame = video.read()
    if ret:
        results = model_pretrained.track(frame, persist=True)
        composed = results[0].plot()
        out.write(composed)
out.release()
video.release()
