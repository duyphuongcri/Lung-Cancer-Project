import numpy as np
import SimpleITK as sitk
import os 
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt 
import cv2 
import torch 

def lung_segment(image, pix_spc):
    mask_air = np.zeros_like(image)
    mask_air[image > -300] = 1.
    y_min, y_max, h_max = image.shape[0], 0, 0
    idx = 0
    n_sl = mask_air.shape[0] # No. slice
    col, row = mask_air.shape[:2]
    mask = np.zeros((mask_air.shape[0], mask_air.shape[2]), np.uint8)
    for slice in range(image.shape[1]//2 -1, image.shape[1]//2 + 1):
        _, img_bw = cv2.threshold((mask_air[:, slice,:]*255.).astype(np.uint8), 150, 255, cv2.THRESH_BINARY_INV)
        _, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            (x,y,w,h) = cv2.boundingRect(contours[i])

            if 5000/pix_spc**2 <= area <= 35000/pix_spc**2 and x+y != 0 and x+w < row \
                    and 70/pix_spc<w<h<=370/pix_spc and h> 200/pix_spc\
                    and  ((n_sl*0.5<y +h//2 < n_sl*0.9 and col>1.5*row) or (h < n_sl and col<=1.5*row)): 
                if y < y_min:
                    y_min = y
                if y+h > y_max:
                    y_max = y+h
                if h > h_max:
                    h_max = h
                    idx = slice
                mask = cv2.drawContours(mask, [contours[i]], -1, (255), -1)
                
    return idx, y_min, y_max, mask

def check_path_dicom(pathCTdicom_, pathSegdicom_, dirName, fileList):
    if (len(fileList) > 1) :
        pathCTdicom_ = dirName
    elif (len(fileList) == 1) and "Segmentation" in dirName:
        pathSegdicom_ = dirName
    
    return pathCTdicom_, pathSegdicom_

def padding_image(image, GT, slice_desired):
    if image.shape[0] <= slice_desired:
        padding = [(slice_desired - image.shape[0],0 ), (0, 0), (0, 0)]
        image_padded = np.pad(image, padding, mode='constant', constant_values=0)
        padding = [(slice_desired - GT.shape[0], 0), (0, 0), (0, 0)]
        GT_padded = np.pad(GT, padding, mode='constant', constant_values=0)
    return image_padded, GT_padded

def load_scan(path):
    
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    #slices.sort(key = lambda x: int(x.InstanceNumber))
    slices.sort(key = lambda x: int(x.SliceLocation))
    # for i in range(len(slices)):
    #     print(slices[i].SliceLocation, slices[i].InstanceNumber)
    try:
        slice_thickness = slices[0].SliceThickness
    except:
        #slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, SliceThickness, PixelSpacingX, PixelSpacingY, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    #spacing = np.array(list(spacing))
    spacing = np.array([SliceThickness, PixelSpacingX, PixelSpacingY], dtype=float)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode="nearest")
    
    return image, new_spacing

def window_normalize_image(image, window_center, window_width):
    """ Window the image to get a specific zone of the image https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    window_image = (window_image - img_min) / window_width
    return window_image

def soft_window_normalize(image, x, y):

    L = np.random.normal(40, x)
    W = abs(np.random.normal(500, y))

    max_threshold = L + W
    min_threshold = L - W

    image[image < min_threshold] = min_threshold
    image[image > max_threshold] = max_threshold
    
    return (image - image.min()) / (image.max() - image.min())

def to_categorical(y, nb_classes):
    """ Convert list of labels to one-hot vectors """
    if len(y.shape) == 2:
        y = y.squeeze(1)

    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)

    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1. # = [1. 0. 0.]
    return ret_mat

def OverallStage_convert(value):
    if isinstance(value, str):
        if value.startswith("1A"):
            return 0.
        if value.startswith("1B"):
            return 1.
        if value.startswith("2A"):
            return 2.
        if value.startswith("2B"):
            return 3.
        if value.startswith("3A"):
            return 4.
        if value.startswith("3B"):
            return 5.
        if value.startswith("4"):
            return 6.
    return float('Nan')

# def OverallStage_convert(value):
#     if isinstance(value, str):
#         if value.startswith("1A"):
#             return 1.
#         if value.startswith("1B"):
#             return 1.
#         if value.startswith("2A"):
#             return 2.
#         if value.startswith("2B"):
#             return 2.
#         if value.startswith("3A"):
#             return 3.
#         if value.startswith("3B"):
#             return 3. 
#         if value.startswith("4"):
#             return 4. 
#     return float('Nan')


# def Tstage_convert(value):
#     if isinstance(value, str):
#         if "1" == value:
#             return 1.
#         if "2" == value:
#             return 2. 
#         if "3" == value:
#             return 3.
#         if "4" ==  value:
#             return 4. 
#     return float('Nan')

def Tstage_convert(value):
    if isinstance(value, str):
        if value.startswith("1a"):
            return 0.
        if value.startswith("1b"):
            return 1.
        if value.startswith("2a"):
            return 2.
        if value.startswith("2b"):
            return 3.
        if value.startswith("3"):
            return 4.
        if value.startswith("4"):
            return 5.
    return float('Nan')

# def Nstage_convert(value):
#     if isinstance(value, str):
#         if "0" == value:
#             return 1.
#         if "1" == value:
#             return 2. 
#         if "2" == value:
#             return 3.
#         if "3" ==  value:
#             return 4. 
#     return float('Nan')

def Nstage_convert(value):
    if isinstance(value, str):
        if "0" == value:
            return 0.
        if "1" == value:
            return 1. 
        if "2" == value:
            return 2.
        if "3" ==  value:
            return 3. 
    return float('Nan')
# def Mstage_convert(value):
#     if isinstance(value, str):
#         if "0" == value:
#             return 1.
#         if "1" == value:
#             return 2. 

#     return float('Nan')

def Mstage_convert(value):
    if isinstance(value, str):
        if "0" == value:
            return 0.
        if "1a" == value:
            return 1.
        if "1b" == value:
            return 2. 

    return float('Nan')

def Histology_convert(value):
    if isinstance(value, str):
        if value.startswith("adenocarcinoma"):
            return 0. 
        if value.startswith("squamous"):
            return 1. 
        if value.startswith("large"):
            return 2. 
        if value.startswith("NOS"):
            return 3. 
    return float('Nan')

def gender_convert(value):
    if isinstance(value, str):
        if value.startswith("male"):
            return 0. 
        if value.startswith("female"):
            return 1.
    return float('Nan')

CONVERTERS = {
    'Overall.stage'   : OverallStage_convert,
    'Clinical.T.Stage': Tstage_convert,
    'Clinical.N.stage': Nstage_convert,
    'Clinical.M.stage': Mstage_convert,
    'Histology'       : Histology_convert,
    'gender'          : gender_convert,
}


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = self.ax.imshow(self.X[:, :, self.ind], cmap="gray")
           
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def sitk_to_np(in_img):
    # type: (sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float]]
    return sitk.GetArrayFromImage(in_img), in_img.GetSpacing()

def imshow_slices(img, mode="axial"):
    
    fig, ax = plt.subplots(1,1)
    plots = []
    if "axial" ==mode:
        for i in range(img.shape[0]):
            plots.append(img[i,:,:])
    elif "coronal"==mode:
        for i in range(img.shape[1]):
            plots.append(img[:,i,:])        
    else:
        for i in range(img.shape[2]):
            plots.append(img[:,:,i])
    y = np.dstack(plots)
    tracker = IndexTracker(ax, y)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def sort_time(x, t, e):
    # Sort
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    t = t[sort_idx]
    e = e[sort_idx]
    return x, t, e
    
def cal_pdf(sur_time, e):
    if torch.is_tensor(sur_time):
        sur_time = sur_time.cpu().detach().numpy()
    objective_prob_u, _ = np.histogram(sur_time[e==1], bins=10, density=True)
    objective_prob_c, _ = np.histogram(sur_time[e==0], bins=10, density=True)
    return objective_prob_u, objective_prob_c