import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def DCF(n):
    length = len(n)
    ll = []
    for i in range(length):
        if i ==0:
            ll.append(n[i])
        elif i == length-1:
            tmp = ll[- 1] + n[i]
            ll.append(tmp)
            return ll
        else:
            tmp = ll[- 1] + n[i]
            ll.append(tmp)

def hist(image):
    Y, U, V = cv.split(image)
    hist_YUV = cv.calcHist([Y], [0], None, [256], [0, 256])
    return hist_YUV
def hist2(image):
    Y, U, V = cv.split(image)
    hist_YUV = cv.calcHist([Y], [0], None, [256], [0, 256])
    return hist_YUV

# γ=0.1, γ=0.3, γ=0.5, γ=2, γ=3, γ=5

image_orig = cv.imread('pic/pic1.jpg')
image_peresvet = cv.imread('pic/peresvet3.jpg')
image_dark = cv.imread('pic/dark.jpg')
image_low_k = cv.imread('pic/low_k.jpg')

cv.imshow("pic's before", np.concatenate((image_orig,image_peresvet,image_dark,image_low_k), axis=1))
h, w = image_low_k.shape[:2]

h, w = image_low_k.shape[:2]

image_orig_YUV = cv.cvtColor(image_orig, cv.COLOR_BGR2YUV)
image_peresvet_YUV = cv.cvtColor(image_peresvet, cv.COLOR_BGR2YUV)
image_dark_YUV = cv.cvtColor(image_dark, cv.COLOR_BGR2YUV)
image_low_k_YUV = cv.cvtColor(image_low_k, cv.COLOR_BGR2YUV)

list_of_image=[]
list_of_name_of_image_before=[]
list_of_name_of_image_after=[]
list_of_plot_of_image=[]
list_of_plot_of_image_after=[]
list_of_name_of_image_before.extend(['hist_orig_YUV_before','hist_peresvet_YUV_before','hist_dark_YUV_before','hist_low_k_YUV_before'])
list_of_plot_of_image.extend([hist(image_orig_YUV),hist(image_peresvet_YUV),hist(image_dark_YUV),hist(image_low_k_YUV)])
list_of_image.extend([image_orig_YUV,image_peresvet_YUV,image_dark_YUV,image_low_k_YUV])

for i in range(len(list_of_name_of_image_before)):
    plt.figure(i+1)
    plt.title(list_of_name_of_image_before[i])
    plt.plot(list_of_plot_of_image[i], '#FF7382')

# for i in range(len(list_of_name_of_image_before)):
#     hist = np.bincount(list_of_image[i].ravel(), minlength=256).tolist()
#     ll = DCF(hist)
#     ll = np.round(np.float32(ll) / (list_of_image[i].shape[0] * list_of_image[i].shape[1]), 4)
#     plt.figure(list_of_name_of_image_before[i]+"DCF", facecolor="lightgray")
#     plt.title("DCF")
#     plt.xlabel("index", fontsize=14)
#     plt.ylabel("value", fontsize=14)
#     plt.tick_params(labelsize=10)
#     plt.grid(linestyle=':')
#     plt.plot(range(0, 256), ll, c='dodgerblue', label=r'$P(rk)=n1+n2+n3+...Nk $')
#     plt.legend()

for i in range(h):
    for j in range(w):
        image_orig_YUV[i][j][0] = ((image_orig_YUV[i][j][0]/255 ) ** 1)*255
        image_peresvet_YUV[i][j][0] = ((image_peresvet_YUV[i][j][0] /255) ** 4)*255
        image_dark_YUV[i][j][0] = ((image_dark_YUV[i][j][0]/255 ) **0.9)*255
        image_low_k_YUV[i][j][0] = ((image_low_k_YUV[i][j][0]/255) ** 2)*255


# cv.imshow("orig_after",cv.cvtColor(image_orig_YUV, cv.COLOR_YUV2BGR))
# cv.imshow("peresvet_after",cv.cvtColor(image_peresvet_YUV, cv.COLOR_YUV2BGR))
# cv.imshow("dark_after",cv.cvtColor(image_dark_YUV, cv.COLOR_YUV2BGR))
# cv.imshow("low_k_after",cv.cvtColor(image_low_k_YUV, cv.COLOR_YUV2BGR))
cv.imshow("pic's_after",np.concatenate((cv.cvtColor(image_orig_YUV, cv.COLOR_YUV2BGR),cv.cvtColor(image_peresvet_YUV, cv.COLOR_YUV2BGR),
                                       cv.cvtColor(image_dark_YUV, cv.COLOR_YUV2BGR), cv.cvtColor(image_low_k_YUV, cv.COLOR_YUV2BGR)), axis=1))


list_of_name_of_image_after.extend(['hist_orig_YUV_after','hist_peresvet_YUV_after','hist_dark_YUV_after','hist_low_k_YUV_after'])
list_of_plot_of_image_after.extend([hist2(image_orig_YUV),hist2(image_peresvet_YUV),hist2(image_dark_YUV),hist2(image_low_k_YUV)])
list_of_image.extend([image_orig_YUV,image_peresvet_YUV,image_dark_YUV,image_low_k_YUV])


for i in range(len(list_of_name_of_image_after)):
    plt.figure(i+6)
    plt.title(list_of_name_of_image_after[i])
    plt.plot(list_of_plot_of_image_after[i], '#75B1FF')

# for i in range(len(list_of_name_of_image_after)):
#     hist = np.bincount(list_of_image[i].ravel(), minlength=256).tolist()
#     ll = DCF(hist)
#     ll = np.round(np.float32(ll) / (list_of_image[i].shape[0] * list_of_image[i].shape[1]), 4)
#     plt.figure(list_of_name_of_image_after[i]+"DCF", facecolor="#BCBCBC")
#     plt.title("DCF2")
#     plt.xlabel("index", fontsize=14)
#     plt.ylabel("value", fontsize=14)
#     plt.tick_params(labelsize=10)
#     plt.grid(linestyle=':')
#     plt.plot(range(0, 256), ll, c='#A362FF', label=r'$P(rk)=n1+n2+n3+...Nk $')
#     plt.legend()

plt.show()


cv.waitKey(0)