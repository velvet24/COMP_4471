from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os

def train():

    model = YOLO("yolo11n-cls.pt")
    results = model.train(data="./datasets/fishNet_split", epochs=20, imgsz=256)

    val_results = model.val()
    print(f"Top-1 Accuracy: {val_results.top1}")
    print(f"Top-5 Accuracy: {val_results.top5}")

    path = "./datasets/FishImgDataset/test/"
    class_names = model.names

    for i in range(len(class_names)):
        image_names = os.listdir(path + class_names[i])
        if not os.path.exists(f"wrong_preds/{class_names[i]}"):
            os.makedirs(f"wrong_preds/{class_names[i]}")
        for image_name in image_names:
            image_path = path + class_names[i] + '/' + image_name
            result = model(image_path)
            pred_label = class_names[result[0].probs.top1]
            prob = result[0].probs.top1conf.item()

            if pred_label != class_names[i]:
                save_path = f"wrong_preds/{class_names[i]}/pred-{pred_label}-{image_name}"
                Image.open(image_path).save(save_path)

if __name__ == "__main__":
    train()