import sys
import os
import re

methods = ["BF", "VSET", "TD", "HIER", "DESC", "PYR"]
lines = {"time":2, "psnr":3, "ssim":4, "vmaf":5, "filterTime":7, "filterPsnr":8, "filterSsim":9, "filterVmaf":10}
metrics = ["VAR", "RANGE", "ERANGE", "MAD"]

with open(sys.argv[2]+"_full.csv", 'w') as outFileFull:
    with open(sys.argv[2]+"_fast.csv", 'w') as outFileFast:
        for metric in metrics:
            outFileFull.write(metric+"\nscene,")
            outFileFast.write(metric+"\nscene,")
            for method in methods:
                for key in lines:
                    outFileFull.write(method+"_"+key+",")
                    outFileFast.write(method+"_"+key+",")
            outFileFull.write("\n")
            outFileFast.write("\n")

            for fileName in os.listdir(sys.argv[1]):
                filePath = os.path.join(sys.argv[1], fileName)
                with open(filePath, 'r') as inFile:
                    outFileFull.write(fileName+",")
                    outFileFast.write(fileName+",")
                    data = inFile.read().rstrip()
                    for method in methods:
                        for startID in [m.start() for m in re.finditer("scan_method:"+method, data)]:
                            linesData = data[startID:].splitlines()
                            if ("scan_metric:"+metric in linesData[0]):
                                for key in lines: 
                                    if ("fast_mode:full" in linesData[0]):
                                        outFileFull.write(linesData[lines[key]]+",") 
                                    elif ("fast_mode:fast" in linesData[0]):
                                        outFileFast.write(linesData[lines[key]]+",") 
                outFileFull.write("\n")
                outFileFast.write("\n")
