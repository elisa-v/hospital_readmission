
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Global configurations
plt.rc('figure', figsize=(6, 4))
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.max_rows', 6)
np.random.seed(12345) # to have the same random numbers