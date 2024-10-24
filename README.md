# Plotting Different Kind of Histogram from Folder of Images


## Commands

### Pixel Histogram of Single Image

```bash
# 1D histogram
python histo.py --img <path to image> --choice 1D

# 2D histogram
python histo.py --img <path to image> --choice 2D

# 3D histogram
python histo.py --img <path to image> --choice 3D
```


### Pixel Histogram of Collection of Images in a Directory

```bash
# 1D histogram
python histo_folder.py --dir <path to directory of images> --choice 1D

# 2D histogram
python histo_folder.py --dir <path to directory of images> --choice 2D

# 3D histogram
python histo_folder.py --dir <path to directory of images> --choice 3D
```


### Feature Histogram of Collection of Images in a Directory

```bash
python feature_histo_folder.py --img_dir <path to directory of images>
```


### Gradient Histogram of Single Image

```bash
python sobel_filter_histo.py --img <path to image> 
```


### Gradient Histogram of Collection of Images in a Directory

```bash
python sobel_filter_histo_folder.py --dir <path to directory of images> 
```


### Sample Results
![Alt Text](dist.png)

