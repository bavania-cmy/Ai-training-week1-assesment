#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Create a 2D array: 4 rows (students) x 3 columns (subjects)
marks = np.array([95,89,90,100,94,95,60,64,58,55,60,58]).reshape(4,3)

print("Shape (rows, cols):", marks.shape)

# 2. Row averages (axis=1) = each student's average
student_avg = np.mean(marks, axis=1)
print("Student averages:", np.round(student_avg, 2))

# 3. Column averages (axis=0) = each subject's average
subject_avg = np.mean(marks, axis=0)
print("Subject averages:", np.round(subject_avg, 2))

# 4. Broadcasting: add 5 grace marks to every score, capped at 100
grace_marks = np.minimum(marks + 5, 100)
print("Marks after +5 grace (capped at 100):\n", grace_marks)

# 5. Weighted score using np.dot (30% Sub1, 30% Sub2, 40% Sub3)
weights = np.array([0.3, 0.3, 0.4])
weighted_scores = np.dot(grace_marks, weights)
print("Weighted scores:", np.round(weighted_scores, 2))

# 6. Measure execution speed using %timeit
get_ipython().run_line_magic('timeit', 'np.dot(grace_marks, weights)')


# In[6]:


# Set up folder paths
BASE_DIR = Path("..").resolve()
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Load your employees.csv
df = pd.read_csv(DATA_DIR / "employees.csv")

# Inspect column types and summary statistics
print("--- INFO ---")
print(df.info())
print("\n--- DESCRIBE ---")
print(df.describe())

# Select by label (loc) and by index position (iloc)
print("\nFirst 3 employees (name & salary):")
print(df.loc[0:2, ["name", "salary"]])

print("\nValue at row 1, column 2:", df.iloc[1, 2])

# Filter: IT employees earning more than 60,000
it_high = df[(df["department"] == "IT") & (df["salary"] > 60000)]
print("\nIT employees with salary > 60,000:")
print(it_high)

