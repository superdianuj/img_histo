import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import os

# source: https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
def histogram_1d(img):
    image = cv2.imread(img)
    print("Image has dimensions: ", image.shape)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("1D Color Histogram")
    plt.xlabel("Color Value")
    plt.ylabel("# of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
        plt.xticks(np.arange(0, 256, step=5))
        plt.grid()
    plt.show()


# source: https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
def histogram_2d(img):
    image = cv2.imread(img)
    print("Image has dimensions: ", image.shape)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    bin_size=32 # 2-256 bins (max 256)

    # create a new figure and then plot a 2D color histogram for the
    # green and blue channels
    fig = plt.figure()
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [bin_size,bin_size],
        [0, 256, 0, 256])   
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for G and B")
    ax.set_xlabel("Green")
    ax.set_ylabel("Blue")
    plt.colorbar(p)
    # plot a 2D color histogram for the green and red channels
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [bin_size,bin_size],
        [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for G and R")
    ax.set_xlabel("Green")
    ax.set_ylabel("Red")
    plt.colorbar(p)
    # plot a 2D color histogram for blue and red channels
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [bin_size,bin_size],
        [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for B and R")
    ax.set_xlabel("Blue")
    ax.set_ylabel("Red")
    plt.colorbar(p)
    # finally, let's examine the dimensionality of one of the 2D
    # histograms
    print("2D histogram shape: {}, with {} values".format(
        hist.shape, hist.flatten().shape[0]))
    
    plt.show()



def histogram_3d(img):
    image = cv2.imread(img)
    print("Image has dimensions: ", image.shape)
    bin_size = 10  # 2-256 bins (max 256)

    # create a 3D color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bin_size, bin_size, bin_size], [0, 256, 0, 256, 0, 256])
    print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

    # Plotting the 3D histogram
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalizing the histogram for better visualization
    hist /= hist.max()

    x = np.arange(0, 256, 256/bin_size)
    y = np.arange(0, 256, 256/bin_size)
    z = np.arange(0, 256, 256/bin_size)

    x, y, z = np.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    c = hist.flatten()

    ax.scatter(x, y, z, c=c, cmap='coolwarm', marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.colorbar(ax.scatter(x, y, z, c=c, cmap='coolwarm', marker='o'))

    plt.title("3D Color Histogram")
    plt.show()


# source: https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/

def histogram_3d_v2(img):
    image = cv2.imread(img)
    print("Image has dimensions: ", image.shape)
    bin_size = 8  # 2-256 bins (max 256)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bin_size, bin_size, bin_size], [0, 256, 0, 256, 0, 256])
    print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))
    
    # initialize our figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ratio = 1000.0
    hist /= hist.max()
    for (b, plane) in enumerate(hist):
        for (g, row) in enumerate(plane):
            for (r, col) in enumerate(row):
                if hist[b][g][r] > 0.0:
                    size = ratio * hist[b][g][r]
                    rgb = (r / (hist.shape[2] - 1),
                           g / (hist.shape[1] - 1),
                           b / (hist.shape[0] - 1))
                    ax.scatter(r, g, b, s=size, facecolors=rgb)
    plt.title("3D Color Histogram")
    plt.xlabel("Red")
    plt.ylabel("Green")
    ax.set_zlabel("Blue")
    plt.show()


def histogram_3d_v3(img):
    # Read the image
    image = cv2.imread(img)
    print("Image has dimensions:", image.shape)
    
    # Convert the image from BGR to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bin_size = 16 # Adjust this value for different levels of detail

    # Calculate the 3D histogram
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [bin_size] * 3, [0, 256] * 3)
    
    # Normalize the histogram
    hist = hist / hist.max()
    
    # Create a meshgrid for plotting
    r, g, b = np.meshgrid(np.linspace(0, 255, bin_size),
                          np.linspace(0, 255, bin_size),
                          np.linspace(0, 255, bin_size))
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D histogram
    scatter = ax.scatter(r, g, b, c=hist, s=hist*1000, alpha=0.6, cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    plt.title("3D Color Histogram")
    
    # Add a color bar
    plt.colorbar(scatter, label='Normalized Frequency')
    plt.show()
    print(f"3D histogram shape: {hist.shape}, with {hist.size} values")
    print(f"Number of non-zero bins: {np.count_nonzero(hist)}")

def main(img,choice):
    if choice=='1d':
        histogram_1d(img)
    elif choice=='2d':
        histogram_2d(img)
    elif choice=='3d':
        histogram_3d_v2(img)
    else:
        print('Invalid choice')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--img', help='image_path')
    parser.add_argument('--choice', help='1d, 2d, 3d',required=True)
    args=parser.parse_args()
    if os.path.exists('img_histo'):
        os.system('rm -r img_histo')
    os.mkdir('img_histo')
    main(args.img,args.choice)

