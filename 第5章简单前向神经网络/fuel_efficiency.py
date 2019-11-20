import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import cross_validate
import skflow

# Read the original dataset
df = pd.read_csv("data/mpg.csv", header=0)
# Convert the displacement column as float
df["displacement"] = df["displacement"].astype(float)
y = df['mpg']
X = df.drop(columns=['mpg'])

plt.figure()