import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    file_path = "Workspace/Dataset/sales_and_customer_insights.csv"
    return pd.read_csv(file_path)

df = load_data()

# Streamlit app title
st.markdown("""<h1 style='color: orange;'>📊 Sales & Customer Insights Dashboard</h1>""", unsafe_allow_html=True)

# Display dataset details
st.subheader("Dataset Overview")
st.write("### Head of the dataset:")
st.write(df.head())
st.write("### Dataset Information:")

st.write(f"Number of Rows: {df.shape[0]}")
st.write(f"Number of Columns: {df.shape[1]}")
st.write(df.describe().loc[['count', 'mean']])



# Top 10 most frequently purchased products
top_products = df.groupby("Product_ID")["Purchase_Frequency"].sum().nlargest(10)

# Seasonal buying trends
seasonal_trends = df.groupby("Season")["Purchase_Frequency"].sum()

# Identifying recurring orders and customer preferences across different categories
customer_preferences = df.groupby("Most_Frequent_Category")["Purchase_Frequency"].sum()

# Visualization: Top 10 Most Frequently Purchased Products
st.subheader("Top 10 Most Frequently Purchased Products")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index, ax=ax, palette="coolwarm")
ax.set_xlabel("Total Purchase Frequency")
ax.set_ylabel("Product ID")
ax.set_title("Top 10 Purchased Products")
ax.set_facecolor('none')
st.pyplot(fig)


# Visualization: Seasonal Buying Trends
st.subheader("Seasonal Buying Trends")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=seasonal_trends.index, y=seasonal_trends.values, marker="o", linestyle="-", linewidth=2.5, color="purple", ax=ax, label='Purchase Frequency')
ax.set_xlabel("Season")
ax.set_ylabel("Total Purchase Frequency")
ax.set_title("Seasonal Trends Over Time")
ax.set_facecolor('none')
ax.legend()
st.pyplot(fig)

# Visualization: Customer Preferences by Category
st.subheader("Customer Preferences Across Different Categories")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=customer_preferences.values, y=customer_preferences.index, ax=ax, palette="magma")
ax.set_xlabel("Total Purchase Frequency")
ax.set_ylabel("Category")
ax.set_title("Customer Preferences Across Categories")
ax.set_facecolor('none')
ax.grid(axis='x', linestyle='--', alpha=0.7)
st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import itertools

# Load dataset
@st.cache_data
def load_data():
    file_path = "Workspace/Dataset/Customer-Segmentation_dataset.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Data Preprocessing
def preprocess_data(df):
    df = df.drop(columns=["Unnamed: 0", "Customer ID", "Review Rating.1"], errors='ignore')
    
    # Identify all categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders if needed later
    
    # Standardizing numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

df_processed = preprocess_data(df)

# Determine optimal clusters using the Elbow Method
def optimal_clusters(data):
    wcss = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), wcss, marker='o', linestyle='--')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method for Optimal K")
    st.pyplot(fig)

# Train K-Means Model
st.sidebar.header("K-Means Clustering")
n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_processed['Cluster'] = kmeans.fit_predict(df_processed)

# Evaluate Clustering Performance
X_train, y_train = df_processed.drop(columns=['Cluster']), df_processed['Cluster']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train)

precision = precision_score(y_train, y_pred, average='weighted')
f1 = f1_score(y_train, y_pred, average='weighted')

# Visualizing Clusters with PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_processed.drop(columns=['Cluster']))
df_processed['Purchase Behavior Dimension'] = principal_components[:, 0]
df_processed['Spending Trend Dimension'] = principal_components[:, 1]

fig, ax = plt.subplots()
sns.scatterplot(data=df_processed, x='Purchase Behavior Dimension', y='Spending Trend Dimension', hue='Cluster', palette='viridis', ax=ax)
ax.set_title("Customer Segmentation Visualization")
st.pyplot(fig)

# Additional Visualization: Distribution of Purchases per Cluster
fig, ax = plt.subplots()
sns.countplot(x=df_processed['Cluster'], palette='viridis', ax=ax)
ax.set_title("Number of Customers per Cluster")
st.pyplot(fig)

# Confusion Matrix Visualization
def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

cm = confusion_matrix(y_train, y_pred)
plot_confusion_matrix(cm, classes=np.unique(y_train))

# Display results
st.write("### Cluster Analysis")
st.write(f"Precision Score: {precision:.2f}")
st.write(f"F1 Score: {f1:.2f}")
