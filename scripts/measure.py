from evaluation import evaluator as eva
from evaluation import basher as bash
import numpy as np
import tempfile
import shutil
import sys
import os
import traceback

binaryPath = "../build/lfFocusMaps"

def makeCmd(inputDir, results, coord, scanMethod, parameter, block, fast, scanRange, distanceOrder, scanMetric, scanSpace, yuvDistance, addressMode):
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
    return command

def run(inputDir, referenceDir, inputRange, outputDir):
    scanMethods = [ ("BF", 16), ("BF", 32), ("BF", 64), ("BF", 128), ("BF", 256)
                    ("RAND", 16), ("RAND", 32), ("RAND", 64)
                    ("HIER", 0), ("HIER", 1), ("DESC", 0), ("DESC", 1), ("SIMP", 0), ("SIMP", 1) ]
    scanMetric = [ "VAR", "RANGE", "IQR", "MAD" ]
    addressModes = [ "WRAP", "CLAMP", "MIRROR", "BORDER", "BLEND" ]
    distanceOrders = [ 1,2,3,4 ]

    workspace = tempfile.mkdtemp()
    resultsPath = os.path.join(workspace, "results")
    tempReferencePath = os.path.join(workspace, "reference")
    shutil.rmtree(resultsPath, ignore_errors=True)
    os.mkdir(resultsPath)
    os.mkdir(tempReferencePath)

    print("Mode, Time [ms], PSNR, SSIM, VMAF")
    for scanMethod in scanMethods:
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
                                    mode = scanMethod[0] + "(" + str(scanMethod[1]) + ")_SS:" + str(scanSpace) + "_" + scanMetric + "_" + blockMode + "_" + fastMode + "_DO:" + str(distanceOrder) + "_" + distanceSpace + "_" + addressMode
                                    for reference in references:
                                        coord = os.path.splitext(reference)[0]
                                        command = makeCmd(inputDir, resultsPath, coord, scanMethod[0], scanMethod[1], block, fast, inputRange,distanceOrder, scanMetric, scapSpace, yuv, addressMode)
                                        result = bash.run(command)
                                        if(result.returncode != 0):
                                            print(result.stderr)
                                            raise Exception("Command not executed.")
                                        r = result.stdout
                                        start = "runs: "
                                        end = " ms"
                                        time += float(r[r.find(start) + len(start):r.rfind(end)])
                                        shutil.copyfile(os.path.join(referenceDir, reference), os.path.join(tempReferencePath, reference))
                                        shutil.copytree(resultsPath, os.path.join(outputDir, mode))
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
                                    print(mode + ", " + str(time) + ", " + str(psnr) + ", " + str(ssim) + ", " + str(vmaf))
    shutil.rmtree(workspace)
try:
    if len(sys.argv) != 5 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Provide arguments for input light field, directory with reference images named x_y (normalized coords, e.g. 0.5_0.5.png), maximum disparity in pixels and output directory: python measure.py ./path/input ./path/reference range ./path/output")
        exit(0)
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

except Exception as e:
    print(e)
    print(traceback.format_exc())
