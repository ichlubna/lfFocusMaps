from evaluation import evaluator as eva
from evaluation import basher as bash
import tempfile
import shutil
import sys
import os
import traceback

binaryPath = "../build/lfFocusMaps"

def makeCmd(inputDir, results, coord, method, parameter, block, fast, scanRange):
    command = binaryPath
    command += " -i "+inputDir
    command += " -c "+coord
    command += " -m "+method
    command += " -p "+str(parameter)
    command += " -o "+results
    command += " -r "+str(scanRange)
    command += " -t 100"
    if block:
        command += " -b "
    if fast:
        command += " -f "
    return command

def loadDir(path):
    files = sorted(os.listdir(path))
    length = Path(files[-1]).stem.split("_")
    self.files = [files[i:i+self.cols] for i in range(0, len(files), self.cols)]

def run(inputDir, referenceDir, inputRange, outputDir):
    methods = [ ("BF", 32) ]

    workspace = tempfile.mkdtemp()
    resultsPath = os.path.join(workspace, "results")
    tempReferencePath = os.path.join(workspace, "reference")
    shutil.rmtree(resultsPath, ignore_errors=True)
    os.mkdir(resultsPath)
    os.mkdir(tempReferencePath)

    print("Mode, Time [ms], PSNR, SSIM, VMAF")
    for method in methods:
        for block in [True, False]:
            for fast in [True, False]:
                references = os.listdir(referenceDir)
                time = 0
                psnr = 0
                ssim = 0
                vmaf = 0
                blockMode = "block" if block else "pixel"
                fastMode = "fast" if fast else "full"
                mode = method[0] + "(" + str(method[1]) + ")_" + blockMode + "_" + fastMode
                for reference in references:
                    coord = os.path.splitext(reference)[0]
                    command = makeCmd(inputDir, resultsPath, coord, method[0], method[1], block, fast, inputRange)
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
