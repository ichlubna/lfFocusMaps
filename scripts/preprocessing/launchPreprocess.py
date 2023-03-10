import traceback
import sys
import preprocess as prep

try:
    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Provide arguments for input light field, output directory and preprocessing method: python preprocess.py ./path/input ./path/output METHOD")
        exit(0)
    prep.preprocess(sys.argv[1], sys.argv[2], sys.argv[3])

except Exception as e:
    print(e)
    print(traceback.format_exc())
