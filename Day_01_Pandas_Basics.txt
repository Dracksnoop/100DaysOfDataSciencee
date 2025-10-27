# 🧠 Day 01 — Pandas Basics

**Date:** 27 October 2025  
**Topic:** Introduction to Pandas Library  
**Goal:** Learn to create and explore data using Pandas.

---

## 🔍 What I Learned Today

- What is **Pandas** and why it’s used in data science.
- How to create a **DataFrame** from a Python dictionary.
- How to explore a dataset using basic functions like:
  - `.head()` → view first few rows
  - `.info()` → check column types and non-null values
  - `.describe()` → quick statistics
- How to access columns and calculate simple statistics:
  - Mean, Median, Mode, etc.
- How to make a **simple bar chart** from a DataFrame.

---

## 🧩 Practice Code

```python
import pandas as pd

# Create a simple DataFrame
data = {'Name': ['Amit', 'Riya', 'Krishna'], 'Age': [21, 24, 22]}
df = pd.DataFrame(data)

# View data
print(df)

# Basic statistics
print("\nAverage Age:", df['Age'].mean())

# Quick plot
df.plot.bar(x='Name', y='Age', title='Ages of People')