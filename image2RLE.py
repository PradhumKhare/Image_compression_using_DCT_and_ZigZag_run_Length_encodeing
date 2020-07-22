import cv2
import numpy as np
import math

# import zigzag functions
from zigzag import *

def get_run_length_encodeing(image):
    i = 0
    skip = 0
    # stream = []
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0] :
        if image[i] != 0 :
            # stream.append((image[i],skip))
            bitstream = bitstream + str(image[i]) + " " + str(skip) + " "
            skip = 0
        else:
            skip += 1
        i += 1
    return bitstream

#defining bloc size
block_size = 8

#Quantization Matrix
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])


#Reading Grayscale image
img = cv2.imread("E:/Image_processing_project/Train_1.jpg",cv2.IMREAD_GRAYSCALE)

[h,w] = img.shape

height = h
width = w

h = np.float32(h)
w = np.float32(w)

# nbh = math.ceil(h/block_size)
# nbh = np.int32(nbh)
#
# nbw = math.ceil(w/block_size)
# nbw = np.int32(nbw)

nbh = h/block_size
nbh = int(nbh) + 1

nbw = w/block_size
nbw = int(nbw) + 1


H = nbh * block_size
W = nbw * block_size


#REQUIRED PADDING
padded_img = np.zeros((H,W))


padded_img[0:height,0:width] = img[0:height,0:width]

cv2.imwrite('uncompressed.bmp',np.uint8(padded_img))


#ENCODING
for i in range(nbh):
    row_start = i * block_size
    row_end   = row_start + block_size
    for j in range(nbw):
        colom_start = j * block_size
        colom_end = colom_start + block_size

        block = padded_img[row_start:row_end,colom_start:colom_end]
        # block = block -  127
        DCT = cv2.dct(block)
        DCT_normalised = np.divide(DCT,QUANTIZATION_MAT).astype(int)

        reordered = zigzag(DCT_normalised)

        reordered = np.reshape(reordered,(block_size,block_size))

        padded_img[row_start:row_end,colom_start:colom_end] = reordered

cv2.imshow('encoded img',np.uint8(padded_img))

arranged = padded_img.flatten()

bitstream = get_run_length_encodeing(arranged)

bitstream = str(padded_img.shape[0]) + " " + str(padded_img.shape[1]) + " " + bitstream + ";"

file1 = open("image.txt","w")
file1.write(bitstream)
file1.close()

cv2.waitKey(0)
cv2.destroyAllWindows()

