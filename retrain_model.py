from ultralytics import YOLO

dataset = 'datasets/data.yaml'

backbone = YOLO("yolov8n.pt")
results = backbone.train(data=dataset, epochs=20)
results = backbone.val()
success = backbone.export(imgsz=640, format='torchscript', optimize=False, half=False, int8=False)
