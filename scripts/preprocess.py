import numpy as np
import sys
import os
import math
import traceback
import cv2

def resize(img, amount):
    return cv2.resize(img, (int(img.shape[1]*amount), int(img.shape[0]*amount)), interpolation=cv2.INTER_LANCZOS4)

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
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                p = img[i,j,c]
                img[i,j,c] = math.sin(frequency*(p/255.0)*math.pi*2)*255
    return img

def run(inputDir, outputDir, method):
    files = sorted(os.listdir(inputDir))
    for file in files:
        inputFile = os.path.join(inputDir, file)
        outputFile = os.path.join(outputDir, file)
        img = cv2.imread(inputFile)
        global result
        if method == "RESIZE_HALF":
            result = resize(img, 0.5)
        elif method == "RESIZE_QUARTER":
            result = resize(img, 0.25)
        elif method == "CONTRAST":
            result = contrast(img, 2)
        elif method == "EDGE":
            result = edge(img, 10)
        elif method == "SHARPEN":
            result = sharpen(img, 9)
        elif method == "EQUAL":
            result = equalize(img)
        elif method == "SINE_FAST":
            result = sine(img, 5)
        elif method == "SINE_SLOW":
            result = sine(img, 2)
        else:
            raise Exception("The requested method is not available.")
        cv2.imwrite(outputFile, result)
try:
    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Provide arguments for input light field, output directory and preprocessing method (CONTRAST, EDGE, SHARPEN, RESIZE_HALF, RESIZE_QUARTER): python preprocess.py ./path/input ./path/output METHOD")
        exit(0)
    run(sys.argv[1], sys.argv[2], sys.argv[3])

except Exception as e:
    print(e)
    print(traceback.format_exc())
