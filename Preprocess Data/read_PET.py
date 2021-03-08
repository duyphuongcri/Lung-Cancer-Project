import pydicom
import os
import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.misc import bytescale
from scipy import ndimage

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

PathDicom = "../2020-lung-cancer/20200510 폐암 예후 예측 (n=317) _radiomics and clinical/PET_20200512/LC00095"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))
        #print(os.path.join(dirName,filename))
# Get ref file
RefDs = pydicom.dcmread(lstFilesDCM[0])
print(RefDs)
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (427, int(RefDs.Rows), int(RefDs.Columns))#(len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.SliceThickness), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))
x = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])

# The array is sized based on 'ConstPixelDims'
image3d = np.zeros(ConstPixelDims, dtype=np.float32)
# Sort file 
lstFilesDCM.sort(key=lambda x:x[-3:] and len(x))

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    
    # if lstFilesDCM.index(filenameDCM) != 87:
    #     continue
    #print(filenameDCM)
    RefDs = pydicom.dcmread(filenameDCM)
    # read the file
    ds = pydicom.read_file(filenameDCM)
    # store the raw image data
    #image3d[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array 
    hu_image = transform_to_hu(RefDs, ds.pixel_array)
    ##Normalize min/max
    lung_image = (hu_image - hu_image.min()) / (hu_image.max() - hu_image.min())
    image3d[lstFilesDCM.index(filenameDCM), :, :] = lung_image

    # plot 3 orthogonal slices
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(hu_image, cmap="gray")
    plt.show()
print(RefDs.PatientName)
print(image3d.shape)