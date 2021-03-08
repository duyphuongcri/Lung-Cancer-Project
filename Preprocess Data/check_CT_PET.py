import os

path_CT = "D:\\LungCancer\\2021-lung-cancer\\Data_label_CT_PET_n_256\\AIDATA_CT_20201105(n=246)_20210202+÷-ñ\\"
path_PET = "D:\\LungCancer\\2021-lung-cancer\Data_label_CT_PET_n_256\\AIDATA_PET_20201105(n=246)_20210202+÷-ñ\\"
print(len(os.listdir(path_PET)), len(os.listdir(path_CT)))
n = 0
max_slice = 0 
min_slice = 1000
for foldername in os.listdir(path_PET):
    n_PET = len(os.listdir(path_PET + foldername))
    # if n_PET == 257:
    #     print(foldername, n_PET)
    n_CT = len(os.listdir(path_CT + "CT_" + foldername))
    if n_PET != n_CT:
        print(foldername, n_PET, n_CT)
    else:
        if n_CT > max_slice:
            max_slice = n_CT
        elif n_CT < min_slice:
            min_slice = n_CT
        #print(foldername, "ok")

    n+=1
print("No. Patients: ", n)
print("Max slice: ", max_slice)
print("Min slice: ", min_slice)