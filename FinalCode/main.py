import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Set display options to show the full pipeline
set_config(display='diagram')

diabetes_df = pd.read_csv(r'C:\Users\begum\Desktop\Projects\Pattern Recognition\FinalCSV\data.csv')
diabetes_df.head()