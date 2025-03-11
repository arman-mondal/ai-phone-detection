from ultralytics import YOLO, checks, hub
checks()

hub.login('b1c20ac18de91bd26e6dc4030f257da8edf75d6856')

model = YOLO('https://hub.ultralytics.com/models/RP7DuFjVSiBE3kWkF8s4')
results = model.train()