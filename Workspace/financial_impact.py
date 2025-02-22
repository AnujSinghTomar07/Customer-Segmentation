
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

d1 = pd.read_csv(r"dataset\shopping_trends.csv")
d2 = pd.read_csv(r"dataset\customer_feedback_satisfaction.csv")
d3 = pd.read_csv(r"dataset\sales_and_customer_insights.csv")

st.markdown("""
    <h1 style="box-shadow:0px 0px 4px 2px gray;margin-bottom:10px;text-align:center;">
            Financial Impact</h1>
""",unsafe_allow_html=True)
st.subheader("Average spending per customer:",divider=True)
ran = st.slider("",0,400,(10,95))
avg = d3[ran[0]:ran[1]] 
line_chart_data = avg[['Customer_ID','Average_Spending_Value']]

st.bar_chart(line_chart_data,x="Customer_ID",y="Average_Spending_Value",color=[.8,.1,.9])

st.divider()

Payment = st.checkbox('Customer Payment Habits',key=1)

lab = ['Credit Card','Venmo','Cash','PayPal','Debit Card','Bank Transfer']

b = d1[['Payment Method']]

if Payment:
    a  = []
    for i in range(len(b.value_counts())):
        a.append(b.value_counts().iloc[i]/3900*100)
    fig, pi = plt.subplots()
    pi.pie(a,labels=lab,colors=s.color_palette("viridis"),autopct='%1.1f%%')
    st.pyplot(fig)

Risky_Customers = st.checkbox('Risky Customers',key=2)

lab = ['Credit Card','Venmo','Cash','PayPal','Debit Card','Bank Transfer']

if Risky_Customers:
    y,cus = d3[["Customer_ID"]],d3[['Churn_Probability','Lifetime_Value','Average_Spending_Value']]
    # b_enc = OneHotEncoder()
    # thr = b_enc.fit_transform([y])
    cus = cus[:400]
    y = y[:400]

    dat = pd.concat([y,cus],axis=1)

    k_me = KMeans(n_clusters=2,random_state=42,n_init=2
    )
    y_pre = k_me.fit(cus,y)
    clust = y_pre.cluster_centers_
    print(y_pre.n_iter_)

    st.scatter_chart(dat,size=20,x="Customer_ID",y='Churn_Probability',x_label="Customer",y_label="Credit Risk")
    # st.scatter_chart(clust,size=15,x_label="Customer",y_label="Credit Risk")
    chr,pl = plt.subplots()
    pl.scatter(dat["Churn_Probability"],dat["Lifetime_Value"],c=y_pre.labels_,cmap="Spectral")
    pl.scatter(clust[:,0],clust[:,1],color="black",marker="X")
    pl.set_ylabel("Customer Income")
    st.pyplot(chr)