import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def max_min(image):
    list = []
    min_max=[]

    for i in range(h):
        for j in range(w):
            list.append(image[i][j][0])

    min_max.extend([max(list),min(list)])
    return min_max

cc=30

def clah(image, name):
    global cc
    y,u,v = cv.split(image)
    print(y)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y=clahe.apply(y)

    new_image = cv.merge((y,u,v))


    plt.figure(cc)
    plt.title(name + "clahe dst")
    plt.hist([y.ravel()], 256, [0, 256], '#62FFBB')
    cc += 1

    return new_image

def hist_before(image):
    axis_x = np.zeros((256), dtype=int)
    axis_y = np.arange(256)
    for i in range(h):
        for j in range(w):
            axis_x[image[i][j][0]] += 1
    points = np.c_[axis_y, axis_x]
    black = np.ones((250, 250, 3), dtype=float)
    hist_hand = cv.polylines(black, [points], False, (255, 0, 0), 1)
    return hist_hand[::-1, :, :]

def dst_image(image, dst):

    h, w = image.shape[:2]

    for i in range(1,h+1):
        for j in range (1,w+1):
            image[i-1][j-1][0]=dst[i-1][j-1]
    return image

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

count=5

def axis_X(image,name):
    global count
    h,w = image.shape[:2]
    axis_x = np.zeros((256), dtype=int)
    axis_y = np.arange(256)
    for i in range(h):
        for j in range(w):
            axis_x[image[i][j][0]] += 1

    s1=[]
    new_image=np.ones((320, 320, 3), dtype=int)
    for i in range(1,h+1):
        for j in range(1,w+1):
            for k in range(image[i-1][j-1][0]):
                s1.append(axis_x[k])
            new_image[i - 1][j - 1][0] = round(((sum(s1) - 1) / (h * w)) * 255)
            new_image[i - 1][j - 1][1]=image[i - 1][j - 1][1]
            new_image[i - 1][j - 1][2] = image[i - 1][j - 1][2]
            s1.clear()

    Y,U,V = cv.split(new_image.astype(np.uint8))



    plt.figure(count)
    plt.title(name+"hand dst")
    plt.hist([Y.ravel()], 256, [0, 256], '#6076FF')
    count+=1

    cv.imshow(name+"hand", cv.cvtColor(new_image.astype(np.uint8), cv.COLOR_YUV2BGR))

    return new_image.astype(np.uint8)


list_of_name_of_image=['orig','peresvet','dark','low_k']

#считывание изображений
image_orig = cv.imread('pic/pic1.jpg')
image_peresvet = cv.imread('pic/peresvet3.jpg')
image_dark = cv.imread('pic/dark.jpg')
image_low_k = cv.imread('pic/low_k.jpg')



#вывод изображений оригинальных
cv.imshow("pic's before", np.concatenate((image_orig,image_peresvet,image_dark,image_low_k), axis=1))

h, w = image_low_k.shape[:2]

#преобразование изображений в модель YUV
image_orig_YUV = cv.cvtColor(image_orig, cv.COLOR_BGR2YUV)
image_peresvet_YUV = cv.cvtColor(image_peresvet, cv.COLOR_BGR2YUV)
image_dark_YUV = cv.cvtColor(image_dark, cv.COLOR_BGR2YUV)
image_low_k_YUV = cv.cvtColor(image_low_k, cv.COLOR_BGR2YUV)


#спилт каналов
Y_orig, U, V = cv.split(image_orig_YUV)
Y_peresvet, U, V = cv.split(image_peresvet_YUV)
Y_dark, U, V = cv.split(image_dark_YUV)
Y_low_k, U, V = cv.split(image_low_k_YUV)

#счетчик фигур
L=20

#гистограмма исходная
list_dst_before_pic=[Y_orig,Y_peresvet, Y_dark, Y_low_k]
list_name_dst_before=['before_orig','before_peresvet','before_dark','before_low_k']
#
#
# for i in range(len(list_dst_before_pic)):
#     plt.figure(L)
#     plt.title(list_name_dst_before[i])
#     plt.hist([list_dst_before_pic[i].ravel()], 256, [0, 256], '#FF7382')
#     L+=1


# for i in range(len(list_dst_before_pic)):
#     hist = np.bincount(list_dst_before_pic[i].ravel(), minlength=256).tolist()
#     ll = DCF(hist)
#     ll = np.round(np.float32(ll) / (list_dst_before_pic[i].shape[0] * list_dst_before_pic[i].shape[1]), 4)
#     plt.figure(list_name_dst_before[i]+" before DCF", facecolor="lightgray")
#     plt.title(list_name_dst_before[i]+"DCF")
#     plt.xlabel("index", fontsize=14)
#     plt.ylabel("value", fontsize=14)
#     plt.tick_params(labelsize=10)
#     plt.grid(linestyle=':')
#     plt.plot(range(0, 256), ll, c='dodgerblue', label=r'$P(rk)=n1+n2+n3+...Nk $')
#     plt.legend()

#выравнивание гистаграмм функцией cv.equalizeHist
dst_orig = cv.equalizeHist(Y_orig)
dst_peresvet = cv.equalizeHist(Y_peresvet)
dst_dark = cv.equalizeHist(Y_dark)
dst_low_k = cv.equalizeHist(Y_low_k)


list_dst = [dst_orig,dst_peresvet,dst_dark,dst_low_k]
list_name_dst=['orig','peresvet','dark','low_k']


#гистограмма после equalize
# for i in range(len(list_dst)):
#     plt.figure(L)
#     plt.title(list_name_dst[i])
#     plt.hist([list_dst[i].ravel()], 256, [0, 256], '#FF7382')
#     L+=1




#
cv.imshow("orig hist_before",hist_before(image_orig_YUV))
cv.imshow("peresvet hist_before",hist_before(image_peresvet_YUV))
cv.imshow("dark hist_before",hist_before(image_dark_YUV))
cv.imshow("low_k hist_before",hist_before(image_low_k_YUV))

# cv.imshow("hist_before", np.concatenate((hist_before(image_orig_YUV),hist_before(image_peresvet_YUV),
#                                          hist_before(image_dark_YUV),hist_before(image_low_k_YUV)), axis=1))

#cv.imshow("after clah", np.concatenate((cv.cvtColor(clah(image_orig_YUV), cv.COLOR_YUV2BGR),cv.cvtColor(clah(image_peresvet_YUV), cv.COLOR_YUV2BGR),
#                                         cv.cvtColor(clah(image_dark_YUV), cv.COLOR_YUV2BGR),cv.cvtColor(clah(image_low_k_YUV), cv.COLOR_YUV2BGR)), axis=1))

cv.imshow("after dst", np.concatenate((cv.cvtColor(dst_image(image_orig_YUV, dst_orig), cv.COLOR_YUV2BGR),
                                         cv.cvtColor(dst_image(image_peresvet_YUV, dst_peresvet), cv.COLOR_YUV2BGR),
                                         cv.cvtColor(dst_image(image_dark_YUV, dst_dark), cv.COLOR_YUV2BGR),
                                         cv.cvtColor(dst_image(image_low_k_YUV, dst_low_k), cv.COLOR_YUV2BGR)), axis=1))

list_of_image_after_dst=[dst_image(image_orig_YUV, dst_orig),dst_image(image_peresvet_YUV, dst_peresvet),
                         dst_image(image_dark_YUV, dst_dark),dst_image(image_low_k_YUV, dst_low_k)]




# for i in range(len(list_name_dst)):
#     hist = np.bincount(list_of_image_after_dst[i].ravel(), minlength=256).tolist()
#     ll = DCF(hist)
#     ll = np.round(np.float32(ll) / (list_of_image_after_dst[i].shape[0] * list_of_image_after_dst[i].shape[1]), 4)
#     plt.figure(list_name_dst[i]+" after dst DCF", facecolor="lightgray")
#     plt.title(list_name_dst[i]+" DCF")
#     plt.xlabel("index", fontsize=14)
#     plt.ylabel("value", fontsize=14)
#     plt.tick_params(labelsize=10)
#     plt.grid(linestyle=':')
#     plt.plot(range(0, 256), ll, c='dodgerblue', label=r'$P(rk)=n1+n2+n3+...Nk $')
#     plt.legend()


# list_of_image_after_clah=[clah(image_orig_YUV, "orig "),clah(image_peresvet_YUV, "peresvet "),
#                           clah(image_dark_YUV, "dark "),clah(image_low_k_YUV, "low_k ")]
#
#
# for i in range(len(list_name_dst)):
#     hist = np.bincount(list_of_image_after_clah[i].ravel(), minlength=256).tolist()
#     ll = DCF(hist)
#     ll = np.round(np.float32(ll) / (list_of_image_after_clah[i].shape[0] * list_of_image_after_clah[i].shape[1]), 4)
#     plt.figure(list_name_dst[i]+" after clah DCF", facecolor="lightgray")
#     plt.title(list_name_dst[i]+ " DCF")
#     plt.xlabel("index", fontsize=14)
#     plt.ylabel("value", fontsize=14)
#     plt.tick_params(labelsize=10)
#     plt.grid(linestyle=':')
#     plt.plot(range(0, 256), ll, c='dodgerblue', label=r'$P(rk)=n1+n2+n3+...Nk $')
#     plt.legend()

orig_hand_alignment=axis_X(image_orig_YUV,"orig")
peresvet_hand_alignment=axis_X(image_peresvet_YUV,"peresvet")
dark_hand_alignment=axis_X(image_dark_YUV,"dark")
low_k_hand_alignment=axis_X(image_low_k_YUV,"low_k")

list_of_image_after_hand_alignment=[orig_hand_alignment,peresvet_hand_alignment,dark_hand_alignment,low_k_hand_alignment]

for i in range(len(list_name_dst)):
    hist = np.bincount(list_of_image_after_hand_alignment[i].ravel(), minlength=256).tolist()
    ll = DCF(hist)
    ll = np.round(np.float32(ll) / (list_of_image_after_hand_alignment[i].shape[0] * list_of_image_after_hand_alignment[i].shape[1]), 4)
    plt.figure(list_name_dst[i]+" after hand alignment DCF", facecolor="lightgray")
    plt.title(list_name_dst[i]+" DCF")
    plt.xlabel("index", fontsize=14)
    plt.ylabel("value", fontsize=14)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(range(0, 256), ll, c='dodgerblue', label=r'$P(rk)=n1+n2+n3+...Nk $')
    plt.legend()

plt.show()
cv.waitKey(0)