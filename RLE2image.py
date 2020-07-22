import cv2
import numpy as np
import math

# import zigzag functions
from zigzag import *

QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

block_size = 8

with open('image.txt','r') as myfile :
    image = myfile.read()

details = image.split(" ")

h = int(details[0])
w = int(details[1])

i = 2
array = np.zeros(h*w).astype(int)
k = 0
while True :

    if (details[i] == ';'):
        break

    k += int(details[i+1])
    array[k] = int(details[i])
    k += 1
    i += 2

array = array.reshape((h,w))

i = 0
j = 0
padded_img = np.zeros((h,w))


while i<h :
    j = 0
    while j < w :
        temp_stream = array[i:i+8,j:j+8]
        block = inverse_zigzag(temp_stream.flatten(),int(block_size),int(block_size))
        de_quantized = np.multiply(block ,  QUANTIZATION_MAT)
        padded_img[i:i+8,j:j+8] = cv2.idct(de_quantized)
        j = j+ 8
    i = i+ 8
padded_img[padded_img > 255] = 255
padded_img[padded_img < 0]  = 0

cv2.imwrite("compressed_image.bmp",np.uint8(padded_img))


