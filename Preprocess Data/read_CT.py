import sys 
sys.path.append("../")
import pydicom
import os
import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.misc import imresize
from scipy import ndimage
import misc

def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

PathDicom = "D:\\LungCancer\\2020-lung-cancer\\collected_data\\CT\\CT_LC00013"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        #if ".dcm" in filename.lower():  # check whether the file's DICOM
        lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = pydicom.dcmread(lstFilesDCM[0])
#print(RefDs)
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (427, int(RefDs.Rows), int(RefDs.Columns))#(len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.SliceThickness), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))
x = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])

# The array is sized based on 'ConstPixelDims'
image3d = np.zeros((len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns)), dtype=np.float32) #np.zeros(ConstPixelDims, dtype=np.float32)

# Sort file 
lstFilesDCM.sort(key=lambda x:x[-3:] and len(x))

# loop through all the DICOM files
plots = []
for filenameDCM in lstFilesDCM:

    #print(filenameDCM)
    RefDs = pydicom.dcmread(filenameDCM)
    # read the file
    ds = pydicom.read_file(filenameDCM)
    pix =ds.pixel_array
    pix[pix==-2000]=0
    hu_image = transform_to_hu(RefDs, pix)
    #lung_image = window_image(hu_image, RefDs.WindowCenter, RefDs.WindowWidth)
    lung_image = misc.window_image(hu_image, 40, 400)
    #resize mage
    #resampled_image = ndimage.interpolation.zoom(lung_image, 128 / lung_image.shape[0], mode='wrap')
    #resampled_image = imresize(lung_image, (128, 128), 'bicubic')
    # Normalize min/max
    #normalized_image = (resampled_image - resampled_image.min()) / (resampled_image.max() - resampled_image.min())
    image3d[lstFilesDCM.index(filenameDCM), :, :] = lung_image#normalized_image

    #####plot 3 orthogonal slices
    # a1 = plt.subplot(1, 2, 1)
    # plt.imshow(ds.pixel_array, cmap="gray")

    # a2 = plt.subplot(1, 2, 2)
    # plt.imshow(resampled_image, cmap="gray")

    hu_image = (hu_image - hu_image.min()) / (hu_image.max() - hu_image.min())
    lung_image = (lung_image - lung_image.min()) / (lung_image.max() - lung_image.min())
    # plt.show()
    plots.append(np.hstack((hu_image, lung_image)))
# for i in range(image3d.shape[1]):
#     plots.append(image3d[:,i,:])

fig, ax = plt.subplots(1, 1)
y = np.dstack(plots)
print(y.shape)
tracker = misc.IndexTracker(ax, y)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

plt.hist(image3d.reshape(-1,1), bins=50)
plt.show()
