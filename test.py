import pydicom
import pydicom_seg
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import misc
import glob, os
import cv2
import scipy.ndimage
import csv 

def lung_segment(image, pix_spc):
    mask_air = np.zeros_like(image)
    mask_air[image > -300] = 1.
    y_min, y_max, h_max = image.shape[0], 0, 0
    idx = 0
    n_sl = mask_air.shape[0] # No. slice
    col, row = mask_air.shape[:2]
    mask = np.zeros((mask_air.shape[0], mask_air.shape[2]), np.uint8)
    for slice in range(image.shape[1]//2 -3, image.shape[1]//2 + 3):
        _, img_bw = cv2.threshold((mask_air[:, slice,:]*255.).astype(np.uint8), 150, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            (x,y,w,h) = cv2.boundingRect(contours[i])
            if 5000/pix_spc**2 <= area <= 35000/pix_spc**2 and x+y != 0 and x+w < row \
                    and 80/pix_spc<w<h<=350/pix_spc \
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

Path = "D:\\LungCancer\\new dataset\\NSCLC-Radiomics"
pathCTdicom = ""
pathSegdicom = ""

list_n_slice = []
size_desired = 128
n_slc_desired = 96
for dirName, subdirList, fileList in os.walk(Path):

    pathCTdicom, pathSegdicom = check_path_dicom(pathCTdicom, pathSegdicom, dirName, fileList)

    if "Lung" in pathCTdicom and "Segmentation" in pathSegdicom:
        if int(dirName[-5:]) < 38:
            pathCTdicom, pathSegdicom = "", ""
            continue
        # if dirName[42:51] + ".npy" in os.listdir("D:\\LungCancer\\new dataset\\dataset\\all\\Imgs"):
        #     pathCTdicom, pathSegdicom = "", ""
        #     continue
        # print(pathCTdicom)
        # print(pathSegdicom)
        patient = misc.load_scan(pathCTdicom)
        patient_pixels = misc.get_pixels_hu(patient)
        pix_spc = patient[0].PixelSpacing[0]*patient_pixels.shape[1]/size_desired
        pix_resampled, _ = misc.resample(patient_pixels, 
                                            patient[0].SliceThickness, 
                                            patient[0].PixelSpacing[0], 
                                            patient[0].PixelSpacing[1], 
                                            [pix_spc,pix_spc,pix_spc])
        # print("Shape before resampling\t", patient_pixels.shape)
        # print("Shape after resampling\t", pix_resampled.shape)

        # segment lung
        idx, y_min, y_max, mask = lung_segment(pix_resampled, pix_spc)
        #print(idx, y_min, y_max, 350/pix_spc)
        if idx==0 or y_max-y_min > 350/pix_spc:
            pathCTdicom, pathSegdicom = "", ""
            print(dirName[42:51], patient_pixels.shape, "-Can not detect lung")
            continue
        # read GT
        dcm = pydicom.dcmread(os.path.join(pathSegdicom,"1-1.dcm"))
        reader = pydicom_seg.SegmentReader()
        result = reader.read(dcm)
        n_len = len(result.available_segments)

        for segment_number in result.available_segments:
            if result.dataset[0x62, 0x02][segment_number-1][0x62, 0x05].value != "Neoplasm, Primary":
                continue
            image = result.segment_image(segment_number)  # lazy construction
            GT_img = misc.sitk_to_np(image)
            GT_resampled, _ = misc.resample(GT_img[0],
                                            patient[0].SliceThickness, 
                                            patient[0].PixelSpacing[0], 
                                            patient[0].PixelSpacing[1],
                                            [pix_spc, pix_spc, pix_spc])
            # GT_resampled = (GT_resampled*255.).astype(np.uint8)
            #GT_resampled = GT_resampled[::-1]
            # print("Shape before resampling\t", GT_img[0].shape)
            # print("Shape after resampling\t", GT_resampled.shape)
                    
            pad = len(GT_resampled) - pix_resampled.shape[0]
            if abs(pad) != 0:
                print(dirName[42:51], patient_pixels.shape, pix_resampled.shape, "-Different shape")
            else:

                #print(dirName[42:51], pix_resampled.shape, GT_resampled.shape, pad)
                assert pix_resampled.shape == GT_resampled.shape, "Dont have the shape"
                

                if pix_resampled.shape[0] > n_slc_desired:
                    bot_y = (y_max + y_min - n_slc_desired)//2
                    if bot_y < 0: 
                        pix_resampled = pix_resampled[0: n_slc_desired, :, :]
                        GT_resampled = GT_resampled[0: n_slc_desired, :, :]
                    elif bot_y + n_slc_desired > pix_resampled.shape[0]:
                        pix_resampled = pix_resampled[-n_slc_desired: , :, :]
                        GT_resampled = GT_resampled[-n_slc_desired: , :, :]
                    else:
                        pix_resampled = pix_resampled[bot_y: bot_y+ n_slc_desired, :, :]
                        GT_resampled = GT_resampled[bot_y: bot_y+n_slc_desired, :, :]                        
                   

                pix_resampled = pix_resampled.astype(np.float32)
                GT_resampled = GT_resampled.astype(np.int)
                pix_resampled = misc.window_normalize_image(pix_resampled, 400, 2000)

                pix_resampled, GT_resampled = misc.padding_image(pix_resampled, GT_resampled, n_slc_desired)
                
                GT_resampled[pix_resampled == 0] = 0

                np.save("D:/LungCancer/new dataset/dataset/all/Imgs/" + dirName[42:51]+ ".npy", pix_resampled)
                np.save("D:/LungCancer/new dataset/dataset/all/GT/" + dirName[42:51]+ ".npy", GT_resampled)


                misc.imshow_slices(np.dstack((pix_resampled, GT_resampled)))


        # reset path
        pathCTdicom, pathSegdicom = "", ""
# f.close()

