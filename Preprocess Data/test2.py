
import numpy as np
import misc
import cv2
import dataloader
import matplotlib.pyplot as plt

img = np.load("D:\\LungCancer\\new dataset\\dataset\\train\\Imgs\\LUNG1-001.npy")
img = misc.soft_window_normalize(img, 100, 100)

img, _ = dataloader.padding_image(img, img, 96)

fig, ax = plt.subplots(1,1)
plots=[]
for i in range(img.shape[0]):
    plots.append(img[i])

y = np.dstack(plots)
tracker = misc.IndexTracker(ax, y)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

