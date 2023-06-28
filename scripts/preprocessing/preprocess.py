import numpy as np
import sys
import os
import math
import cv2
import random
import cython

def resize(img, amount):
    return cv2.resize(img, (int(img.shape[1]*amount), int(img.shape[0]*amount)), interpolation=cv2.INTER_AREA)

def brightNoise(img):
    alpha = 1+0.5*(random.random()-0.5)
    beta = 1+0.5*(random.random()-0.5)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def contrast(img, amount):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=amount, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def edge(img, amount):
    #edges = cv2.Canny(img,amount,amount*4)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(img, cv2.CV_8U)
    #edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    #img = cv2.addWeighted(img, 1, edges, 1, 0.0)
    return cv2.addWeighted(img, 1, edges, 1, 0.0)

def sharpen(img, amount):
    kernel = np.array([[-1,-1,-1],
                       [-1, amount,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def equalize(img):
    yuvImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb);
    (Y, U, V) = cv2.split(yuvImg)
    Y = cv2.equalizeHist(Y)
    U = cv2.equalizeHist(U)
    V = cv2.equalizeHist(V)
    yuvImg = cv2.merge([Y, U, V])
    return cv2.cvtColor(yuvImg, cv2.COLOR_YCrCb2BGR);

def sine(img, frequency):
    rows,cols,channels = img.shape
    yuvImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb);
    originalImg = yuvImg.copy()
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
            #for c in range(1):
                p = yuvImg[i,j,c]
                yuvImg[i,j,c] = (math.sin(frequency*(p/255.0)*math.pi*2)*0.5+0.5)*255
    img = cv2.addWeighted(yuvImg, 0.5, originalImg, 0.5, 0)
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR);

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def median(img):
    return cv2.medianBlur(img,3)

def bilateral(img):
    return cv2.bilateralFilter(img, 16, 8, 16)

def gaussian(img,amount):
    return cv2.GaussianBlur(img,(amount, amount),0)

def clamp(val, minval, maxval):
    if val < minval: return minval
    if val > maxval: return maxval
    return val

def highlight(img):
    BLOCK_HALF = 1
    BLOCK_SIZE = (2*BLOCK_HALF+1)*(2*BLOCK_HALF+1)
    h = img.shape[0]
    w = img.shape[1]
    result = img.copy()
    for y in range(h):
        for x in range(w):
            pixels = []
            mean = np.array([0.0,0.0,0.0])
            for i in range(-BLOCK_HALF, BLOCK_HALF+1):
                for j in range(-BLOCK_HALF, BLOCK_HALF+1):
                    yy = clamp(y+i, 0, h-1)
                    xx = clamp(x+j, 0, w-1)
                    px = img[yy, xx]
                    pixels.append(px)
                    mean += px
            mean /= BLOCK_SIZE
            maxDist = -1
            color = None
            for i in range(BLOCK_SIZE):
                d = (mean[0] - pixels[i][0])**2 + (mean[1] - pixels[i][1])**2 + (mean[2] - pixels[i][2])**2
                if d > maxDist:
                    maxDist = d
                    color = pixels[i]
            result[y,x] = color
    return result

def preprocess(inputDir, outputDir, method):
    files = sorted(os.listdir(inputDir))
    for file in files:
        inputFile = os.path.join(inputDir, file)
        outputFile = os.path.join(outputDir, file)
        img = cv2.imread(inputFile)
        global result
        if method == "RESIZE_EIGHTH":
            result = resize(img, 0.125);
        elif method == "RESIZE_QUARTER":
            result = resize(img, 0.25)
        elif method == "GAUSSIAN_LIGHT_HALF":
            result = gaussian(img, 5)
            result = resize(result, 0.5)
        elif method == "GAUSSIAN_HEAVY":
            result = gaussian(img, 19)
        elif method == "GAUSSIAN_ULTRA_HEAVY":
            result = gaussian(img, 41)
        elif method == "CONTRAST":
            result = contrast(img, 2)
        elif method == "EDGE":
            result = edge(img, 10)
        elif method == "SHARPEN":
            result = sharpen(img, 9)
        elif method == "EQUAL":
            result = equalize(img)
        elif method == "SINE_FAST":
            result = sine(img, 3)
        elif method == "SINE_SLOW":
            result = sine(img, 1)
        elif method == "DENOISE":
            result = denoise(img)
        elif method == "MEDIAN":
            result = median(img)
        elif method == "BILATERAL":
            result = bilateral(img)
        elif method == "HIGHLIGHT":
            result = highlight(img)
        elif method == "BRIGHT_NOISE":
            result = brightNoise(img)
        elif method == "CHAIN":
            result = denoise(img)
            result = sine(img, 1)
            result = highlight(result)
        else:
            raise Exception("The requested method is not available.")
        cv2.imwrite(outputFile, result)
