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

def makeCmd(inputDir, results, coord, scanMethod, parameter, blockSize, fast, scanRange, distanceOrder, scanMetric, scanSpace, colorDist, addressMode, useSecondary, mapFilter, gridAspect):
    command = binaryPath
    command += " -i "+inputDir
    command += " -c "+coord
    command += " -m "+scanMethod
    command += " -s "+str(scanSpace)
    command += " -e "+scanMetric
    command += " -d "+str(distanceOrder)
    command += " -p "+str(parameter)
    command += " -o "+results
    command += " -r "+str(scanRange)
    command += " -a "+addressMode
    command += " -t 100"
    command += " -y "+colorDist
    command += " -z "+mapFilter
    command += " -g "+gridAspect
    command += " -b "+str(blockSize)
    if fast:
        command += " -f "
    if useSecondary:
        command += " -l "
    command += " -n";
    return command

class Comparison:
    PSNR = 0
    SSIM = 0
    VMAF = 0
    measurements = 0

    def compareResults(self, referenceFile, resultFile, workspace):
        comparisonReferencePath = os.path.join(workspace, "compReference")
        comparisonResultPath = os.path.join(workspace, "compResult")
        os.mkdir(comparisonResultPath)
        os.mkdir(comparisonReferencePath)

        shutil.copyfile(referenceFile, os.path.join(comparisonReferencePath, os.path.basename(referenceFile)))
        shutil.copyfile(resultFile, os.path.join(comparisonResultPath, os.path.basename(resultFile)))
        evaluator = eva.Evaluator()
        metrics = evaluator.metrics(comparisonReferencePath, comparisonResultPath)
        self.PSNR += float(metrics.psnr)
        self.SSIM += float(metrics.ssim)
        self.VMAF += float(metrics.vmaf)
        self.measurements += 1

        shutil.rmtree(comparisonResultPath)
        shutil.rmtree(comparisonReferencePath)

    def psnr(self):
        return self.PSNR/self.measurements

    def ssim(self):
        return self.SSIM/self.measurements

    def vmaf(self):
        return self.VMAF/self.measurements

def run(inputDir, referenceDir, inputRange, gridWidth, gridHeight, gridAspect):
    scanMethods = [ ("BF", 32), ("BF", 64)\
                    ("BFET", 32), ("BFET", 64)\
                    ("VS", 32), ("VSET", 32),\
                    ("RAND"), ("TD"),\
                    ("HIER", 0), ("DESC", 0),\
                    ("PYR", 0), ("PYR", 0), ("PYR", 0), ("PYR", 0), ("PYR", 0) ]
    scanMetrics = [ "VAR", "ERANGE", "RANGE", "MAD" ]
    addressModes = [ "WRAP", "CLAMP", "MIRROR", "BORDER", "BLEND", "ALTER" ]
    preprocesses = [ "NONE", "CONTRAST", "EDGE", "SHARPEN", "EQUAL", "SINE_FAST", "SINE_SLOW", "DENOISE", "MEDIAN", "BILATERAL", "HIGHLIGHT"]
    pyramidPreprocess = [ "RESIZE_QUARTER", "RESIZE_EIGHTH", "GAUSSIAN_ULTRA_HEAVY", "GAUSSIAN_HEAVY", "GAUSSIAN_LIGHT"]
    filters = [ "MED", "SNN", "KUW" ]
    distanceOrders = [ 1,2,3,4 ]

    workspace = tempfile.mkdtemp()
    inputPath = os.path.join(workspace, "input")
    shutil.copytree(inputDir, inputPath)
    downPath = inputPath+"_down"
    secondaryPath = inputPath+"_sec"
    resultsPath = os.path.join(workspace, "results")
    shutil.rmtree(resultsPath, ignore_errors=True)
    os.mkdir(resultsPath)
    filteredResultName = os.path.join(resultsPath, "renderImagePostFiltered.png")
    rawResultName = os.path.join(resultsPath, "renderImage.png")

    pyramidID = 0
    useSecondary = False
    for preprocess in preprocesses:
        if preprocess ==  "NONE":
            useSecondary = False
        else:
            prepr.preprocess(inputDir, secondaryPath, preprocess)
            useSecondary = True
        for scanMethod in scanMethods:
            pyramidMode = ""
            if str(scanMethod[0]) == "PYR":
                pyramidMode = pyramidPreprocess[pyramidID]
                prepr.preprocess(inputDir, downPath, pyramidMode)
                pyramidID +=1
            for addressMode in addressModes:
                for scanSpace in np.linspace(0.5,3,21):
                    for scanMetric in scanMetrics:
                        for distanceOrder in distanceOrders:
                            for blockSize in np.linspace(0,20,41):
                                for mapFilter in filters:
                                    for fast in [True, False]:
                                        for colorDist in ["RGB", "YUV", "Y", "YUVw"]:
                                            references = os.listdir(referenceDir)
                                            fastMode = "fast" if fast else "full"
                                            if pyramidMode != "":
                                                pyramidMode = "_"+pyramidMode
                                            mode =  "scan_method:"      + scanMethod[0]+pyramidMode  + "|" +\
                                                    "scan_parameter:"   + str(scanMethod[1])    + "|" +\
                                                    "scan_space:"       + str(scanSpace)        + "|" +\
                                                    "scan_metric:"      + scanMetric            + "|" +\
                                                    "preprocessing:"    + preprocess            + "|" +\
                                                    "block_size:"       + str(blockSize)        + "|" +\
                                                    "fast_mode:"        + fastMode              + "|" +\
                                                    "distance_order:"   + str(distanceOrder)    + "|" +\
                                                    "color_distance:"   + colorDist             + "|" +\
                                                    "address_mode:"     + addressMode           + "|" +\
                                                    "map_filter:"       + mapFilter

                                            rawComparison = Comparison()
                                            filteredComparison = Comparison()

                                            time = 0
                                            timeFilter = 0
                                            for reference in references:
                                                coord = os.path.splitext(reference)[0]
                                                separate  = coord.split("_")
                                                normalized = str(float(separate[1])/(gridHeight-1))+"_"+str(float(separate[0])/(gridWidth-1))
                                                command = makeCmd(inputPath, resultsPath, normalized, scanMethod[0], scanMethod[1], blockSize, fast, inputRange, distanceOrder, scanMetric, scanSpace, colorDist, addressMode, useSecondary, mapFilter, gridAspect)
                                                result = bash.run(command)
                                                if(result.returncode != 0):
                                                    print(result.stderr)
                                                    print(result.stdout)
                                                    print("Used command: "+command)
                                                    raise Exception("Command not executed.")

                                                r = result.stdout
                                                time += float(''.join(r.split("runs of interpolation: ")[1].split(" ms")[0]))
                                                timeFilter += float(''.join(r.split("runs of map filtering: ")[1].split(" ms")[0]))

                                                referenceName = os.path.join(referenceDir, reference)
                                                shutil.copy(rawResultName, "./"+reference)
                                                rawComparison.compareResults(referenceName, rawResultName, workspace)
                                                filteredComparison.compareResults(referenceName, filteredResultName, workspace)

                                            time /= len(references)
                                            timeFilter /= len(references)
                                            print(   mode + "\n" +\
                                                    "raw \n"+\
                                                    "time: " + str(time) + " ms \n" +\
                                                    "psnr: " + str(rawComparison.psnr()) + "\n"   +\
                                                    "ssim: " + str(rawComparison.ssim()) + "\n"   +\
                                                    "vmaf: " + str(rawComparison.vmaf()) + "\n" +\
                                                    "filtered \n"+
                                                    "filtering time: " + str(timeFilter) + " ms \n" +\
                                                    "psnr: " + str(filteredComparison.psnr()) + "\n"   +\
                                                    "ssim: " + str(filteredComparison.ssim()) + "\n"   +\
                                                    "vmaf: " + str(filteredComparison.vmaf()) + "\n")
    shutil.rmtree(workspace)
try:
    if len(sys.argv) != 7 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Provide arguments for input light field, directory with reference images named x_y, focus range start_end, grid width and height: python measure.py ./path/input ./path/reference min_max W H gridAspect")
        exit(0)
    run(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])

except Exception as e:
    print(e)
    print(traceback.format_exc())
