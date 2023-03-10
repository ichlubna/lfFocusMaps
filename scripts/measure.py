from evaluation import evaluator as eva
from evaluation import basher as bash
from preprocessing import preprocess as prepr
import numpy as np
import tempfile
import shutil
import sys
import os
import traceback

binaryPath = "../build/lfFocusMaps"

def makeCmd(inputDir, results, coord, scanMethod, parameter, block, fast, scanRange, distanceOrder, scanMetric, scanSpace, yuvDistance, addressMode, useSecondary):
    command = binaryPath
    command += " -i "+inputDir
    command += " -c "+coord
    command += " -m "+scanMethod
    command += " -s "+scanSpace
    command += " -e "+scanMetric
    command += " -d "+str(distanceOrder)
    command += " -p "+str(parameter)
    command += " -o "+results
    command += " -r "+str(scanRange)
    command += " -a "+addressMode
    command += " -t 100"
    if block:
        command += " -b "
    if fast:
        command += " -f "
    if yuvDistance:
        command += " -y "
    if useSecondary:
        command += " -l "
    return command

def run(inputDir, referenceDir, inputRange):
    scanMethods = [ ("BF", 16), ("BF", 32), ("BF", 64), ("BF", 128), ("BF", 256)
                    ("RAND", 16), ("RAND", 32), ("RAND", 64)
                    ("HIER", 0), ("HIER", 1), ("DESC", 0), ("DESC", 1), ("PYR", 0), ("PYR", 1), ("PYR", 0), ("PYR", 1) ]
    scanMetric = [ "VAR", "RANGE", "IQR", "MAD" ]
    addressModes = [ "WRAP", "CLAMP", "MIRROR", "BORDER", "BLEND" ]
    preprocesses = [ "NONE", "CONTRAST", "EDGE", "SHARPEN", "EQUAL", "SINE_FAST", "SINE_SLOW", "DENOISE", "MEDIAN", "BILATERAL"]
    distanceOrders = [ 1,2,3,4 ]

    workspace = tempfile.mkdtemp()
    inputPath = os.path.join(workspace, "input")
    shutil.copytree(inputDir, inputPath)
    downPath = inputPath+"_down"
    secondaryPath = inputPath+"_sec"
    resultsPath = os.path.join(workspace, "results")
    tempReferencePath = os.path.join(workspace, "reference")
    shutil.rmtree(resultsPath, ignore_errors=True)
    os.mkdir(resultsPath)
    os.mkdir(tempReferencePath)

    print("Mode, Time [ms], PSNR, SSIM, VMAF")
    pyramidID = 0
    useSecondary = False
    for preprocess in preprocesses:
        if preprocess ==  "NONE":
            useSecondary = False
        else:
            prepr.preprocess(inputDir, secondaryPath, preprocess)
            useSecondary = True
        for scanMethod in scanMethods:
            if scanMethod == "PYR":
                if pyramidID < 2:
                    prepr.preprocess(inputDir, downPath, "RESIZE_HALF")
                else:
                    prepr.preprocess(inputDir, downPath, "RESIZE_QUARTER")
                pyramidID +=1
            for addressMode in addressModes:
                for scanSpace in np.linspace(0.5,3,30)
                    for scanMetric in scanMetrics:
                        for distanceOrder in distanceOrders:
                            for block in [True, False]:
                                for fast in [True, False]:
                                    for yuv in [True, False]:
                                        references = os.listdir(referenceDir)
                                        time = 0
                                        psnr = 0
                                        ssim = 0
                                        vmaf = 0
                                        blockMode = "block" if block else "pixel"
                                        fastMode = "fast" if fast else "full"
                                        distanceSpace = "yuv" if yuv else "rgb"
                                        mode =  "scan_method:"      + scanMethod[0]         + "|" +
                                                "scan_parameter:"   + str(scanMethod[1])    + "|" +
                                                "scan_space:"       + str(scanSpace)        + "|" +
                                                "scan_metric:"      + scanMetric            + "|" +
                                                "preprocessing:"    + preprocess            + "|" +
                                                "block_mode:"       + blockMode             + "|" +
                                                "fast_mode:"        + fastMode              + "|" +
                                                "distance_order:"   + str(distanceOrder)    + "|" +
                                                "distance_space:"   + distanceSpace         + "|" +
                                                "address_mode:"     + addressMode
                                        for reference in references:
                                            coord = os.path.splitext(reference)[0]
                                            command = makeCmd(inputPath, resultsPath, coord, scanMethod[0], scanMethod[1], block, fast, inputRange,distanceOrder, scanMetric, scapSpace, yuv, addressMode, useSecondary)
                                            result = bash.run(command)
                                            if(result.returncode != 0):
                                                print(result.stderr)
                                                raise Exception("Command not executed.")
                                            r = result.stdout
                                            start = "runs: "
                                            end = " ms"
                                            time += float(r[r.find(start) + len(start):r.rfind(end)])
                                            shutil.copyfile(os.path.join(referenceDir, reference), os.path.join(tempReferencePath, reference))
                                            os.remove(os.path.join(resultsPath, "focusMap.png"))
                                            evaluator = eva.Evaluator()
                                            metrics = evaluator.metrics(tempReferencePath, resultsPath)
                                            psnr += float(metrics.psnr)
                                            ssim += float(metrics.ssim)
                                            vmaf += float(metrics.vmaf)
                                        time /= len(references)
                                        psnr /= len(references)
                                        ssim /= len(references)
                                        vmaf /= len(references)
                                        print(  mode + "\n" +
                                                "time:" + str(time) + "ms|" +
                                                "psnr:" + str(psnr) + "|"   +
                                                "ssim:" + str(ssim) + "|"   +
                                                "vmaf:" + str(vmaf))
    shutil.rmtree(workspace)
try:
    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Provide arguments for input light field, directory with reference images named x_y (normalized coords, e.g. 0.5_0.5.png), maximum normalized disparity: python measure.py ./path/input ./path/reference range")
        exit(0)
    run(sys.argv[1], sys.argv[2], sys.argv[3])

except Exception as e:
    print(e)
    print(traceback.format_exc())
