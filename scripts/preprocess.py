import sys
import os
import traceback

def loadDir(path):
    files = sorted(os.listdir(path))
    length = Path(files[-1]).stem.split("_")
    self.files = [files[i:i+self.cols] for i in range(0, len(files), self.cols)]

def run(inputDir, outputDir, method):

try:
    if len(sys.argv) != 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Provide arguments for input light field, output directory and preprocessing method (CONTRAST, EDGE, SHARPEN, PAD-ZERO, PAD-CYCLE, PAD-REPEAT, PAD-BLUR): python preprocess.py ./path/input ./path/output METHOD")
        exit(0)
    run(sys.argv[1], sys.argv[2], sys.argv[3])

except Exception as e:
    print(e)
    print(traceback.format_exc())
