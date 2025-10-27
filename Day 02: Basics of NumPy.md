# 🔢 Day 02: Basics of NumPy

## 🎯 Introduction

Welcome to **Day 02** of my **100 Days of Data Science** challenge! Today, I explored **NumPy** (Numerical Python), the fundamental package for scientific computing in Python. NumPy provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. It's the backbone of data science libraries like Pandas, Scikit-learn, and TensorFlow.

## 🎓 Learning Goals

- Understand what NumPy is and why it's essential for data science
- Learn to create and manipulate NumPy arrays
- Practice array indexing, slicing, and reshaping
- Explore mathematical operations and broadcasting
- Work with random number generation and statistical functions

## 🔍 Concepts Practiced

✅ Creating NumPy arrays (1D, 2D, 3D)  
✅ Array attributes (`.shape`, `.dtype`, `.ndim`, `.size`)  
✅ Indexing and slicing arrays  
✅ Array reshaping and transposing  
✅ Mathematical operations (element-wise and aggregate)  
✅ Broadcasting principles  
✅ Random number generation with `np.random`  
✅ Statistical functions (`.mean()`, `.std()`, `.sum()`, `.min()`, `.max()`)  

## 💻 Code Snippets
```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
arr3 = np.zeros((3, 3))
arr4 = np.ones((2, 4))
arr5 = np.linspace(0, 1, 5)  # 5 values between 0 and 1

# Array attributes
print(arr1.shape)  # (5,)
print(arr1.dtype)  # int64
print(arr1.ndim)   # 1

# Reshaping
matrix = np.arange(12).reshape(3, 4)
print(matrix.T)  # Transpose

# Mathematical operations
arr = np.array([1, 2, 3, 4])
print(arr * 2)  # Element-wise multiplication
print(arr.sum())  # 10
print(arr.mean())  # 2.5

# Random arrays
random_arr = np.random.randint(1, 100, size=(3, 3))
print(random_arr)

# Statistical operations
data = np.array([23, 45, 67, 89, 12, 34, 56])
print(f"Mean: {data.mean()}")
print(f"Std Dev: {data.std()}")
print(f"Max: {data.max()}")
```

## 📝 Observations & Learnings

- **NumPy arrays are faster** than Python lists because they store data in contiguous memory blocks
- Array operations are **vectorized**, meaning no need for explicit loops — operations apply to entire arrays
- **Broadcasting** allows NumPy to perform operations on arrays of different shapes automatically
- `.reshape()` doesn't change the data, just how it's viewed — great for transforming data structures
- Random number generation is crucial for simulations, machine learning initialization, and data sampling
- Statistical functions work along specified axes, making it easy to compute row-wise or column-wise statistics

## 💭 Reflection

Today's session opened my eyes to **why NumPy is so powerful** for numerical computing. The speed and efficiency of array operations compared to Python lists is remarkable. The most challenging part was understanding **broadcasting rules** — when arrays of different shapes can be combined and when they can't. I spent extra time experimenting with different array shapes to grasp this concept. The connection between NumPy and yesterday's Pandas lesson became clear: Pandas DataFrames are built on top of NumPy arrays!

## 🚀 Next Steps

Tomorrow, I'll dive into **Data Cleaning with Pandas** to learn how to handle missing values, duplicates, and data preprocessing techniques.

👉 [Day 03: Data Cleaning with Pandas](../Day-03/README.md)

---

> 💡 Part of my 100 Days of Data Science learning journey.

> 📂 [← Day 01: Basics of Pandas](../Day-01/README.md)