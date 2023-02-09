import subprocess
import shlex

def run(command, workingDir="./"):
        result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=workingDir)
        return result
