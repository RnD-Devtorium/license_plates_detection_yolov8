import cv2 as cv
from glob import glob
from ultralytics import YOLO

videos = glob('inputs/*.mp4')

np_model = YOLO('runs/detect/train3/weights/best.pt')

video = cv.VideoCapture(videos[0])
ret, frame = video.read()

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./outputs/video_result2.avi', fourcc, 20.0, size)

ret = True

while ret:
    ret, frame = video.read()
    if ret:
        results = np_model.track(frame, persist=True)
        composed = results[0].plot()
        out.write(composed)
out.release()
video.release()
