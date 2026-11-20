import sys
import pandas, numpy, sklearn, matplotlib, seaborn
print("Python:", sys.version.split()[0])
for lib in [pandas, numpy, sklearn, matplotlib, seaborn]:
    print(lib.__name__, lib.__version__)
print("Setup OK")