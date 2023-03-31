import os
import sys
import shutil
import traceback

class Renamer:
    inputPath = ""
    outputPath = ""
    direction = [0,0]
    inputGrid = [0,0]
    split = True
    rowMajor = True

    def printHelpAndExit(self):
        print("This script renames arbitrary input lightfield files into the desired format.")
        print("Run as: python renameLFImages inputPath outputPath directionX directionY gridWidth gridHeight split")
        print("inputPath and outputPath - paths to the existing directories")
        print("directionX and directionY - how is LF captured - directionX = 0 means left to right, directionY = 0 means top to bottom and 1 the other way")
        print("rowMajor - 1 means that LF was captured row by row, 0 column by column")
        print("gridWidth and gridHeight - number of images in the grid in X and Y axis")
        print("split - if set to 1 splits the data into two datasets for measurement with reference middle images, e.g. splits 15x15 grid to 8x8 and rest between images")
        exit(0)

    def checkArgs(self):
        ARGS_NUM = 8
        if len(sys.argv) < ARGS_NUM+1  or sys.argv[1] == "-h" or sys.argv[1] == "--help":
            self.printHelpAndExit()
        self.inputPath = sys.argv[1]
        self.outputPath = sys.argv[2]
        self.direction[0] = int(sys.argv[3])
        self.direction[1] = int(sys.argv[4])
        self.inputGrid[0] = int(sys.argv[5])
        self.inputGrid[1] = int(sys.argv[6])
        self.rowMajor = int(sys.argv[7]) == 1
        self.split = int(sys.argv[8]) == 1

    def getCoords(self, imgID):
        inputCol = imgID % self.inputGrid[0]
        inputRow = imgID // self.inputGrid[0]
        if not self.rowMajor:
            inputCol = imgID // self.inputGrid[1]
            inputRow = imgID % self.inputGrid[1]
        col = 0
        row = 0
        if self.direction[0] == 0:
            col = inputCol
        else:
            col = self.inputGrid[0]-inputCol-1
        if self.direction[1] == 0:
            row = inputRow
        else:
            row = self.inputGrid[1]-inputRow-1
        return (col, row)

    def getName(self, colRow, grid):
       colLength = len(str(grid[0]))
       rowLength = len(str(grid[1]))
       return str(colRow[0]).zfill(colLength)+"_"+str(colRow[1]).zfill(rowLength)

    def convert(self):
        files = sorted(os.listdir(self.inputPath))
        imgID = 0
        for file in files:
           inFile = self.inputPath
           inFile = os.path.join(inFile, file)
           extension = os.path.splitext(file)[1]
           coords = self.getCoords(imgID)
           name = self.getName(coords, self.inputGrid)+extension
           outFile = self.outputPath
           outFile = os.path.join(outFile, name)
           shutil.copyfile(inFile, outFile)
           imgID += 1
    
    def convertSplit(self):
        files = sorted(os.listdir(self.inputPath))
        imgID = 0
        data = os.path.join(self.outputPath, "data")
        reference = os.path.join(self.outputPath, "reference")
        os.mkdir(data)
        os.mkdir(reference)
        for file in files:
           inFile = self.inputPath
           inFile = os.path.join(inFile, file)
           extension = os.path.splitext(file)[1]
           inputCoords = self.getCoords(imgID)
           coords = (0,0)
           isData = True
           if (inputCoords[0] % 2) == 0 and (inputCoords[1] % 2) == 0:
               coords = (inputCoords[0] // 2, inputCoords[1] // 2)
               isData = True
           else:
               coords = (inputCoords[0]/2.0, inputCoords[1]/2.0)
               isData = False
           name = self.getName(coords, self.inputGrid)+extension
           outFile = ""
           if isData:
               outFile = os.path.join(data, name)
           else:
               outFile = os.path.join(reference, name)
           shutil.copyfile(inFile, outFile)
           imgID += 1

    def run(self):
        self.checkArgs()
        if self.split:
            self.convertSplit()
        else:
            self.convert()

try:
    renamer = Renamer()
    renamer.run()
except Exception as e:
    print(e)
    print(traceback.format_exc())

