import sys
import pickle
import flask
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import imblearn

def print_library_versions():
    print("=" * 55)
    print("Python Environment Information")
    print("=" * 55)

    print(f"Python          : {sys.version.split()[0]}")
    print(f"Flask           : {flask.__version__}")
    print(f"NumPy           : {np.__version__}")
    print(f"Pandas          : {pd.__version__}")
    print(f"Scikit-learn    : {sklearn.__version__}")
    print(f"Matplotlib      : {matplotlib.__version__}")
    print(f"Seaborn         : {sns.__version__}")
    print(f"Imbalanced-learn: {imblearn.__version__}")
    print(f"Pickle (Format) : {pickle.format_version}")

    print("=" * 55)

print_library_versions()