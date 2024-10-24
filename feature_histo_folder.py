import os
from PIL import Image
import torch
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
import argparse

def histogram_1d(dir):
    img_names = os.listdir(dir)
    img_paths = [os.path.join(dir, img) for img in img_names]
    features_collector = []

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE)

    for img in img_paths:
        image_real = Image.open(img).convert("RGB")
        outputs = pipe(image_real)
        features = np.array(outputs[0][0])
        features_collector.append(np.array(features))

    features_collector = np.array(features_collector)
    print("Features shape: ", features_collector.shape)

    # Flatten the features and compute the histogram
    flattened_features = features_collector.flatten()
    plt.figure()
    plt.title("1D Histogram of Extracted Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.hist(flattened_features, bins=64, color='blue', alpha=0.7)
    plt.grid(True)
    # plt.show()
    plt.savefig(f'histogram_1d_{dir}.png')
    plt.close()

def main(img_dir):
    histogram_1d(img_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing images")
    args = parser.parse_args()
    main(args.img_dir)