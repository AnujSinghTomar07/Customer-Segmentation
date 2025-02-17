import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def customer_loyalty_analysis():
    st.title("Customer Loyalty Analysis")
    
    # Load customer segmentation dataset
    file_path_1 = r"C:\Users\Dell\Infotact\customer_segmentation\dataset\customer_feedback_satisfaction.csv"
    df1 = load_data(file_path_1)
    
    # Count occurrences of each loyalty level
    loyalty_counts = df1['LoyaltyLevel'].value_counts()
    
    # Display top values for highest loyalty level
    highest_loyalty = df1[df1['LoyaltyLevel'] == loyalty_counts.idxmax()]
    st.write("Customers with Highest Loyalty Level:")
    st.dataframe(highest_loyalty.head())
    
    # Define valid colors
    colors = ['#FFD700', '#CD7F32', '#C0C0C0']  # Gold, Silver, Bronze
    
    # Plot pie chart
    st.subheader("Customer Loyalty Level Distribution")
    fig, ax = plt.subplots()
    ax.pie(loyalty_counts, labels=loyalty_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig)

def high_value_customers_analysis():
    st.title("High-Value Customers in Loyalty Program")
    
    # Load sales and customer insights dataset
    file_path_2 = r"C:\Users\Dell\Infotact\customer_segmentation\dataset\sales_and_customer_insights.csv"
    df2 = load_data(file_path_2)
    
    # Filter high-value customers participating in Loyalty Program
    loyal_customers = df2[df2['Retention_Strategy'] == 'Loyalty Program']
    
    # Display all High-Value Customers
    st.subheader("All High-Value Customers in Loyalty Program")
    st.dataframe(loyal_customers[['Customer_ID', 'Lifetime_Value', 'Purchase_Frequency']])
    
    # Histogram for Lifetime Value Distribution
    st.subheader("Lifetime High Value Distribution of Loyalty Program Customers")
    fig, ax = plt.subplots()
    ax.hist(loyal_customers['Lifetime_Value'], bins=20, color='green', edgecolor='black')
    ax.set_xlabel("Lifetime Value")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Distribution of Lifetime Value for Loyalty Program Participants")
    st.pyplot(fig)

    st.subheader("Purchase Frequency Data for Loyalty Program Customers")

    # Sort loyal customers by Purchase Frequency in descending order
    sorted_loyal_customers = loyal_customers[['Customer_ID', 'Purchase_Frequency']].sort_values(by='Purchase_Frequency', ascending=False)

    # Reset index to clean up the table
    sorted_loyal_customers = sorted_loyal_customers.reset_index(drop=True)

    # Display the sorted dataframe
    st.dataframe(sorted_loyal_customers)


    
    # Bar chart for Purchase Frequency
    st.subheader("Purchase Frequency of Loyalty Program Customers")
    fig, ax = plt.subplots()
    loyal_customers['Purchase_Frequency'].value_counts().sort_index().plot(kind='bar', ax=ax, color='grey', edgecolor='black')
    ax.set_xlabel("Purchase Frequency")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Purchase Frequency Distribution for Loyalty Program Participants")
    st.pyplot(fig)

if __name__ == "__main__":
    customer_loyalty_analysis()
    high_value_customers_analysis()






# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# def main():
#     st.title("Customer Loyalty Level Distribution")
    
#     # Load the dataset
#     file_path = r"C:\Users\Dell\Infotact\customer_segmentation\dataset\customer_feedback_satisfaction.csv"  # Ensure this file is in the correct directory
#     df = pd.read_csv(file_path)
    
#     # Count occurrences of each loyalty level
#     loyalty_counts = df['LoyaltyLevel'].value_counts()

    
#     # Display top values for highest loyalty level
#     highest_loyalty = df[df['LoyaltyLevel'] == loyalty_counts.idxmax()]
#     st.write("Customers with Highest Loyalty Level:")
#     st.dataframe(highest_loyalty.head())
    
#     # Define valid colors
#     colors = ['#FFD700', '#CD7F32', '#C0C0C0']  # Gold, Silver, Bronze
    
#     # Plot pie chart
#     fig, ax = plt.subplots()
#     ax.pie(loyalty_counts, labels=loyalty_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
#     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
#     st.pyplot(fig)


# if __name__ == "__main__":
#     main()





# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_data(file_path):
#     return pd.read_csv(file_path)

# def customer_loyalty_analysis():
#     st.title("Customer Loyalty Analysis")
    
#     # Load customer segmentation dataset
#     file_path_1 = r"C:\Users\Dell\Infotact\customer_segmentation\dataset\customer_feedback_satisfaction.csv"
#     df1 = load_data(file_path_1)
    
#     # Count occurrences of each loyalty level
#     loyalty_counts = df1['LoyaltyLevel'].value_counts()
    
#     # Display top values for highest loyalty level
#     highest_loyalty = df1[df1['LoyaltyLevel'] == loyalty_counts.idxmax()]
#     st.write("Customers with Highest Loyalty Level:")
#     st.dataframe(highest_loyalty.head())
    
#     # Define valid colors
#     colors = ['#FFD700', '#CD7F32', '#C0C0C0']  # Gold, Silver, Bronze
    
#     # Plot pie chart
#     st.subheader("Customer Loyalty Level Distribution")
#     fig, ax = plt.subplots()
#     ax.pie(loyalty_counts, labels=loyalty_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
#     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
#     st.pyplot(fig)

# def high_value_customers_analysis():
#     st.title("High-Value Customers in Loyalty Program")
    
#     # Load customer feedback satisfaction dataset
#     file_path_2 = r"C:\Users\Dell\Infotact\customer_segmentation\dataset\sales_and_customer_insights.csv"
#     df2 = load_data(file_path_2)
    
#     # Box Plot for Satisfaction Score Distribution
#     st.subheader("Satisfaction Score Distribution of Loyalty Program Customers")
#     fig, ax = plt.subplots()
#     sns.boxplot(y=df2['SatisfactionScore'], ax=ax, color='blue')
#     ax.set_ylabel("Satisfaction Score")
#     ax.set_title("Box Plot of Satisfaction Score for Loyalty Program Participants")
#     st.pyplot(fig)
    
#     # Scatter Plot for Purchase Frequency vs. Satisfaction Score
#     st.subheader("Purchase Frequency vs. Satisfaction Score")
#     fig, ax = plt.subplots()
#     sns.scatterplot(x=df2['Purchase_Frequency'], y=df2['SatisfactionScore'], ax=ax, color='green')
#     ax.set_xlabel("Purchase Frequency")
#     ax.set_ylabel("Satisfaction Score")
#     ax.set_title("Scatter Plot of Purchase Frequency vs. Satisfaction Score")
#     st.pyplot(fig)
    
#     st.write("These visualizations help in identifying customer satisfaction levels and their engagement patterns.")

# if __name__ == "__main__":
#     customer_loyalty_analysis()
#     high_value_customers_analysis()



# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_data(file_path):
#     return pd.read_csv(file_path)

# def customer_loyalty_analysis():
#     st.title("Customer Loyalty Analysis")
    
#     # Load customer feedback satisfaction dataset
#     file_path = r"C:\Users\Dell\Infotact\customer_segmentation\dataset\customer_feedback_satisfaction.csv"
#     df = load_data(file_path)
    
#     # Ensure Satisfaction Score is numeric
#     df['SatisfactionScore'] = pd.to_numeric(df['SatisfactionScore'], errors='coerce')
    
#     # Categorize satisfaction scores
#     df['Category'] = 'Medium Satisfaction'
#     df.loc[df['SatisfactionScore'] <= 40, 'Category'] = 'Low Satisfaction'
#     df.loc[df['SatisfactionScore'] > 80, 'Category'] = 'High Satisfaction'
    
#     # Sort dataframe after categorization
#     df_sorted = df.sort_values(by='SatisfactionScore', ascending=False).reset_index()
    
#     # Count occurrences of each loyalty level
#     loyalty_counts = df['LoyaltyLevel'].value_counts()
    
#     # Display top values for highest loyalty level
#     highest_loyalty = df[df['LoyaltyLevel'] == loyalty_counts.idxmax()]
#     st.write("Customers with Highest Loyalty Level:")
#     st.dataframe(highest_loyalty.head())
    
#     # Pie Chart for Loyalty Level Distribution
#     st.subheader("Customer Loyalty Level Distribution")
#     fig, ax = plt.subplots()
#     ax.pie(loyalty_counts, labels=loyalty_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FFD700', '#CD7F32', '#C0C0C0'])
#     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
#     st.pyplot(fig)
    
#     # Display some satisfied customers
#     st.subheader("Some Satisfied Customers")
#     satisfied_customers = df[df['SatisfactionScore'] >= df['SatisfactionScore'].quantile(0.9)]
#     st.dataframe(satisfied_customers[['CustomerID','Age', 'Gender','PurchaseFrequency', 'LoyaltyLevel','SatisfactionScore']].head())
    
#     # Scatter Plot for Customers with Categorized Coloring
#     st.subheader("Scatter Plot of Customer Satisfaction Scores")
    
#     category_colors = {
        
#         'High Satisfaction': 'green',
#         'Medium Satisfaction': 'yellow',
#         'Low Satisfaction': 'red'
#     }
    
#     fig, ax = plt.subplots()
#     for category, color in category_colors.items():
#         subset = df_sorted[df_sorted['Category'] == category]
#         ax.scatter(subset['CustomerID'], subset['SatisfactionScore'], color=color, label=category, s=10)
    
#     ax.set_xlabel("Customer ID")
#     ax.set_ylabel("Satisfaction Score")
#     ax.set_title("Scatter Plot of Customer Satisfaction Scores with Categorization")
#     ax.legend()
#     st.pyplot(fig)
    
    
# if __name__ == "__main__":
#     customer_loyalty_analysis()
