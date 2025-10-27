# 📊 Day 01: Basics of Pandas

## 🎯 Introduction

Today marks the beginning of my **100 Days of Data Science** challenge! I started with **Pandas**, one of the most essential libraries for data manipulation and analysis in Python. Pandas provides powerful data structures like DataFrames and Series, making it incredibly easy to work with structured data, perform exploratory data analysis (EDA), and prepare datasets for machine learning models.

## 🎓 Learning Goals

- Understand what Pandas is and its role in data science
- Learn to create and manipulate DataFrames
- Practice data indexing and selection techniques
- Explore basic statistical functions
- Visualize data using Pandas plotting capabilities

## 🔍 Concepts Practiced

✅ Creating DataFrames from dictionaries and lists  
✅ Indexing and selecting data (`.loc`, `.iloc`)  
✅ Using `.head()`, `.tail()`, and `.shape`  
✅ Statistical summaries with `.mean()`, `.describe()`, `.info()`  
✅ Basic data visualization with `.plot()`  
✅ Handling missing data basics  

## 💻 Code Snippets
```python
import pandas as pd

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Score': [85, 90, 78, 92]
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())
print(df.describe())
print(df['Age'].mean())

# Indexing
print(df.loc[0])  # First row
print(df.iloc[:, 1])  # Second column

# Simple plot
df.plot(x='Name', y='Score', kind='bar')
```

## 📝 Observations & Learnings

- **DataFrames** are like Excel sheets in Python — rows and columns with labels make data manipulation intuitive
- The `.describe()` function instantly provides statistical insights (mean, std, min, max, quartiles)
- `.info()` is super useful for checking data types and identifying null values
- Pandas integrates seamlessly with Matplotlib for quick visualizations
- Indexing with `.loc` (label-based) vs `.iloc` (position-based) requires attention to avoid confusion

## 💭 Reflection

What fascinated me most today was how **Pandas simplifies complex data operations** into just a few lines of code. Tasks that would take dozens of lines in pure Python become single-function calls. The biggest challenge was understanding the difference between `.loc` and `.iloc` — I initially mixed them up, but practice clarified the distinction. I'm excited to dive deeper into data cleaning and manipulation techniques!

## 🚀 Next Steps

Tomorrow, I'll be exploring **NumPy Basics** to understand the foundation of numerical computing in Python.

👉 [Day 02: NumPy Basics](https://github.com/Dracksnoop/100DaysOfDataSciencee/blob/main/Day%2002%3A%20Basics%20of%20NumPy.md)

---

> 💡 Part of my 100 Days of Data Science learning journey.