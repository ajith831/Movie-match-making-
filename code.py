import pandas as pd

# -------------------------------------------
# Data Collection
# -------------------------------------------
# The dataset "Tamil_movies_dataset.csv" is assumed to be collected from a public online source such as IMDb or Wikipedia using web scraping
# tools (e.g., BeautifulSoup or Selenium), or downloaded directly if available as an open dataset.
# Alternatively, APIs like TMDb API could be used for real-time data access.

# Load the dataset
df = pd.read_csv("Tamil_movies_dataset.csv")

# -------------------------------------------
# Data Cleaning
# -------------------------------------------
# Handling missing values
df.fillna("Unknown", inplace=True)

# Removing duplicates
df.drop_duplicates(inplace=True)

# Standardizing column names (optional)
df.columns = [col.strip().capitalize() for col in df.columns]

# Preview available columns
print("Available columns:", df.columns.tolist())

# Preview unique genres
if 'Genre' in df.columns:
    print("\nAvailable genres in the dataset:")
    print(df['Genre'].dropna().unique())

# -------------------------------------------
# Get user input for genre
# -------------------------------------------
genre_input = input("\nEnter a genre to filter movies by (e.g., Action, Drama): ")

# Filter the dataset by genre (case-insensitive match)
filtered_df = df[df['Genre'].str.contains(genre_input, case=False, na=False)]

# Show results
if not filtered_df.empty:
    print(f"\nMovies in genre '{genre_input}':\n")
    print(filtered_df)
else:
    print(f"\nNo movies found in genre '{genre_input}'.")

# -------------------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------------------
# Techniques may include:
# - Value counts for genres, directors, ratings, years
# - Distribution plots of ratings or release years
# - Correlation heatmaps (if numerical features exist)
# Tools: pandas, matplotlib, seaborn

# Example (optional, comment/uncomment to use):
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.countplot(data=df, x='Genre')
# plt.xticks(rotation=45)
# plt.title("Genre Distribution")
# plt.show()

# -------------------------------------------
# Feature Engineering
# -------------------------------------------
# Examples:
# - Extract year from release date
# - Create binary features like "Is_Hit" based on rating or revenue
# - Encode categorical features using one-hot or label encoding for modeling

# -------------------------------------------
# Model Building
# -------------------------------------------
# If using machine learning (e.g., predicting movie success or rating):
# - Models: Decision Trees, Random Forest, Logistic Regression, Naive Bayes, etc.
# - These models are suitable due to their simplicity, interpretability, and performance on structured data.

# -------------------------------------------
# Model Evaluation
# -------------------------------------------
# Metrics:
# - Accuracy, Precision, Recall, F1 Score (classification)
# - MAE, RMSE (regression)
# - Cross-validation for robust performance estimates

# -------------------------------------------
# Visualization & Interpretation
# -------------------------------------------
# Key findings and patterns will be presented using:
# - Bar charts, pie charts, histograms
# - Heatmaps, scatter plots
# - Dashboards (optional using tools like Plotly Dash or Streamlit)

# -------------------------------------------
# Deployment
# -------------------------------------------
# The project can be deployed as:
# - A web application using Streamlit or Flask
# - An interactive dashboard using Power BI/Tableau
# - Or shared via a Jupyter Notebook on GitHub or Kaggle
