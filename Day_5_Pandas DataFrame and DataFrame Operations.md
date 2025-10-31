# Pandas DataFrame and Operations Guide

## What is a Pandas DataFrame?

A **DataFrame** is a 2-dimensional labeled data structure in pandas, similar to a spreadsheet or SQL table. It consists of rows and columns where each column can have different data types (integers, floats, strings, etc.).

## Creating DataFrames

```python
import pandas as pd

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)

# From list of lists
data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df = pd.DataFrame(data, columns=['Name', 'Age'])

# From CSV file
df = pd.read_csv('file.csv')

# From Excel file
df = pd.read_excel('file.xlsx')
```

## Basic DataFrame Operations

### Viewing Data

```python
df.head()          # First 5 rows
df.tail()          # Last 5 rows
df.shape           # Dimensions (rows, columns)
df.info()          # Data types and memory usage
df.describe()      # Statistical summary
df.columns         # Column names
df.index           # Row indices
```

### Selecting Data

```python
# Select single column
df['Name']
df.Name

# Select multiple columns
df[['Name', 'Age']]

# Select rows by index
df.loc[0]          # By label
df.iloc[0]         # By position

# Select rows and columns
df.loc[0:2, 'Name':'Age']
df.iloc[0:3, 0:2]

# Conditional selection
df[df['Age'] > 25]
df[(df['Age'] > 25) & (df['City'] == 'Paris')]
```

## Data Manipulation Operations

### Adding/Removing Columns

```python
# Add new column
df['Salary'] = [50000, 60000, 70000]
df['Bonus'] = df['Salary'] * 0.1

# Remove column
df.drop('Bonus', axis=1, inplace=True)
df.drop(columns=['Bonus'])
```

### Adding/Removing Rows

```python
# Add row
new_row = {'Name': 'David', 'Age': 28, 'City': 'Berlin'}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Remove row
df.drop(0, axis=0, inplace=True)  # Drop row at index 0
```

### Sorting

```python
# Sort by column
df.sort_values('Age')                    # Ascending
df.sort_values('Age', ascending=False)   # Descending
df.sort_values(['City', 'Age'])          # Multiple columns
```

### Filtering

```python
# Filter rows
df[df['Age'] > 30]
df[df['City'].isin(['Paris', 'London'])]
df[df['Name'].str.startswith('A')]
```

## Data Cleaning Operations

### Handling Missing Values

```python
# Check for missing values
df.isnull()
df.isnull().sum()

# Fill missing values
df.fillna(0)
df.fillna({'Age': 0, 'City': 'Unknown'})
df.fillna(method='ffill')  # Forward fill

# Drop missing values
df.dropna()                # Drop rows with any NaN
df.dropna(axis=1)          # Drop columns with any NaN
df.dropna(subset=['Age'])  # Drop rows where Age is NaN
```

### Renaming

```python
# Rename columns
df.rename(columns={'Name': 'Full_Name', 'Age': 'Years'})

# Rename index
df.rename(index={0: 'first', 1: 'second'})
```

### Data Type Conversion

```python
df['Age'] = df['Age'].astype(int)
df['City'] = df['City'].astype(str)
df['Date'] = pd.to_datetime(df['Date'])
```

## Aggregation Operations

### GroupBy

```python
# Group by single column
df.groupby('City').mean()
df.groupby('City')['Age'].sum()

# Group by multiple columns
df.groupby(['City', 'Department']).count()

# Multiple aggregations
df.groupby('City').agg({
    'Age': ['mean', 'min', 'max'],
    'Salary': 'sum'
})
```

### Statistical Operations

```python
df['Age'].mean()      # Average
df['Age'].median()    # Median
df['Age'].sum()       # Sum
df['Age'].min()       # Minimum
df['Age'].max()       # Maximum
df['Age'].std()       # Standard deviation
df['Age'].count()     # Count non-null values
df['Age'].value_counts()  # Frequency of each value
```

## Merging and Joining

```python
# Merge two DataFrames
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Salary': [50000, 60000]})

# Inner join
pd.merge(df1, df2, on='ID')

# Left join
pd.merge(df1, df2, on='ID', how='left')

# Concatenate DataFrames
pd.concat([df1, df2], axis=0)  # Vertically
pd.concat([df1, df2], axis=1)  # Horizontally
```

## Apply Functions

```python
# Apply function to column
df['Age'] = df['Age'].apply(lambda x: x + 1)

# Apply function to multiple columns
df[['Age', 'Salary']] = df[['Age', 'Salary']].apply(lambda x: x * 1.1)

# Apply function row-wise
df['Total'] = df.apply(lambda row: row['Age'] + row['Salary'], axis=1)
```

## Exporting Data

```python
# To CSV
df.to_csv('output.csv', index=False)

# To Excel
df.to_excel('output.xlsx', index=False)

# To JSON
df.to_json('output.json')

# To dictionary
df.to_dict()
```

## Best Practices

1. **Always use `inplace=True` carefully** - it modifies the original DataFrame
2. **Chain operations** for cleaner code when possible
3. **Use vectorized operations** instead of loops for better performance
4. **Check data types** after loading data with `df.dtypes`
5. **Handle missing values** before performing analysis
6. **Use meaningful column names** without spaces

## Common Use Cases

- **Data Analysis**: Exploring patterns and trends
- **Data Cleaning**: Removing duplicates, handling missing values
- **Data Transformation**: Reshaping, pivoting, aggregating
- **Data Visualization**: Preparing data for plotting
- **Machine Learning**: Feature engineering and preprocessing