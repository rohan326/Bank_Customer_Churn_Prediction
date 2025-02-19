import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv(r"C:\Users\91944\Desktop\Data_Analyst_Project\Bank Customer Churn Prediction.csv")
#(df.to_string())

print(df.info())  # Column types and missing values
print(df.describe())  # Summary statistics
print(df.head()) 


# Set style
plt.style.use("ggplot")

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Histogram of Credit Scores
axes[0, 0].hist(df["credit_score"], bins=20, color="red", edgecolor="black")
axes[0, 0].set_title("Distribution of Credit Scores")
axes[0, 0].set_xlabel("Credit Score")
axes[0, 0].set_ylabel("Frequency")

# 2. Churn Rate (Pie Chart)
churn_counts = df["churn"].value_counts()
axes[0, 1].pie(churn_counts, labels=["Stayed", "Churned"], autopct="%1.1f%%", colors=["green", "red"], startangle=140)
axes[0, 1].set_title("Churn Rate")

# 3. Age vs. Balance (Scatter Plot)
axes[0, 2].scatter(df["age"], df["balance"], color="purple", alpha=0.5)
axes[0, 2].set_title("Age vs. Balance")
axes[0, 2].set_xlabel("Age")
axes[0, 2].set_ylabel("Balance")

# 4. Number of Customers by Country (Bar Chart)
country_counts = df["country"].value_counts()
axes[1, 0].bar(country_counts.index, country_counts.values, color=["blue", "orange", "green"])
axes[1, 0].set_title("Number of Customers by Country")
axes[1, 0].set_xlabel("Country")
axes[1, 0].set_ylabel("Count")

# 5. Churn Rate by Age Group (Bar Chart)
df["age_group"] = pd.cut(df["age"], bins=[18, 30, 40, 50, 60, 100], labels=["18-30", "30-40", "40-50", "50-60", "60+"])
churn_by_age = df.groupby("age_group")["churn"].mean() * 100
axes[1, 1].bar(churn_by_age.index, churn_by_age.values, color="red")
axes[1, 1].set_title("Churn Rate by Age Group")
axes[1, 1].set_xlabel("Age Group")
axes[1, 1].set_ylabel("Churn Rate (%)")

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
