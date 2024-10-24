import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def histogram_1d(img):
    image = img
    print("Image has dimensions: ", image.shape)
    max_val = np.max(image)
    print("Max value in image: ", max_val)
    min_val = np.min(image)
    print("Min value in image: ", min_val)
    img_flat = image.flatten()
    plt.figure()
    plt.xlabel("Value", fontsize=20)
    plt.ylabel("# of Pixels",fontsize=20)
    plt.hist(img_flat, bins=64, color='blue', alpha=0.7)
    plt.xlim([0, 256])
    plt.xticks(np.arange(min_val-1, max_val+1, step=16))
    plt.show()


def sobelize(src,scale=1,delta=0,ddepth=cv2.CV_16S):
    src = cv2.GaussianBlur(src, (2*scale+1, 2*scale+1), 0)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def main(dirr):
    img_names = os.listdir(dirr)
    img_paths = [os.path.join(dirr, img) for img in img_names]
    total_sobelizer=[]
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img=cv2.resize(img,(1024,1024))
        sobelizer=sobelize(img)
        total_sobelizer.append(sobelizer)

    total_sobelizer=np.concatenate(total_sobelizer,axis=0)
    histogram_1d(total_sobelizer)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args=parser.parse_args()
    main(args.dir)
