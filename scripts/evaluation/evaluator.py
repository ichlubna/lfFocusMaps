import tempfile
import os
import shutil
from evaluation import basher

class Metrics:
    ssim = 0
    psnr = 0
    vmaf = 0
    vif = [0,0,0,0]

    def avgVif(self):
        return (sum(self.vif)/len(self.vif))

    def __str__(self):
        return "PSNR: "+str(self.psnr)+"\nSSIM: "+str(self.ssim)+"\nVMAF: "+str(self.vmaf)+"\nVIF: "+str(self.avgVif())

class Evaluator:
    tmpDir = ""
    ffmpegPath = "ffmpeg"
    internalNumberingWidth = 4

    def checkAndParsePath(self, path):
        if not os.path.exists(path):
            raise Exception("Path "+path+" does not exist!")
        if not os.path.isdir(path):
            raise Exception("Path "+path+" is not a directory!")
        files = sorted(os.listdir(path))
        if len(files) == 0:
            raise Exception("The path "+path+" is empty!")
        extension = os.path.splitext(files[0])[1]
        return files, extension

    def prepareDirs(self, originalDir, distortedDir):
        originalFiles, originalExtension = self.checkAndParsePath(originalDir)
        distortedFiles, distortedExtension = self.checkAndParsePath(distortedDir)
        if len(originalFiles) != len(distortedFiles):
            raise Exception("The input directories do not contain the same number of images!")
        tmpOriginal = os.path.join(self.tmpDir,"original")
        tmpDistorted = os.path.join(self.tmpDir,"distorted")
        os.mkdir(tmpOriginal)
        os.mkdir(tmpDistorted)
        for i in range(0,len(originalFiles)):
           shutil.copyfile(os.path.join(originalDir,originalFiles[i]), os.path.join(tmpOriginal,str(i).zfill(self.internalNumberingWidth)+originalExtension))
           shutil.copyfile(os.path.join(distortedDir,distortedFiles[i]), os.path.join(tmpDistorted,str(i).zfill(self.internalNumberingWidth)+distortedExtension))
        return tmpOriginal, tmpDistorted, originalExtension, distortedExtension

    def metrics(self, inputOriginalDir, inputDistortedDir):
        m = Metrics()
        originalDir, distortedDir, originalExtension, distortedExtension = self.prepareDirs(inputOriginalDir, inputDistortedDir)

        commandStart = self.ffmpegPath+" -i "+os.path.join(originalDir,"%0"+str(self.internalNumberingWidth)+"d"+originalExtension)+" -i "+os.path.join(distortedDir,"%0"+str(self.internalNumberingWidth)+"d"+distortedExtension)
        commandEnd = "-f null -"

        result = basher.run(commandStart+" -lavfi ssim "+commandEnd)
        m.ssim = result.stderr.partition("All:")[2]
        m.ssim = m.ssim.partition(" ")[0]

        result = basher.run(commandStart+" -lavfi psnr "+commandEnd)
        m.psnr = result.stderr.partition("average:")[2]
        m.psnr = m.psnr.partition(" min")[0]

        result = basher.run(commandStart+" -lavfi libvmaf "+commandEnd)
        m.vmaf = result.stderr.partition("VMAF score: ")[2]
        m.vmaf = m.vmaf.partition("\n")[0]

        result = basher.run(commandStart+" -lavfi vif "+commandEnd)
        for i in range(0,4):
            m.vif[i] = result.stderr.partition("scale="+str(i)+" average:")[2]
            m.vif[i] = float(m.vif[i].partition(" min:")[0])

        shutil.rmtree(originalDir)
        shutil.rmtree(distortedDir)
        return m

    def __init__(self):
        self.tmpDir = os.path.join(tempfile.mkdtemp(), '')
