import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st



st.set_page_config(page_title='Predict House Price', page_icon='house')

df = pd.read_csv("house_data.csv")

st.write("""
### Predicting housing price in Seattle area.
""")
st.write("""
This data is taken from Kaggle: [[DATA](https://www.kaggle.com/harlfoxem/housesalesprediction)]
""")

st.write("""
In this project we are going to use Neural Networks to predict the price of the house. It's a 5 layers Neural Network and you can see hows easy is to build one. It's only took me 7 lines of code to do it and you can see the code by clicking on button below.
""")

if st.button('Model'):
    st.image("project/img/codesnippet.jpg", caption='Sequential Model')

st.write("""
Here you can see some usefull charts by your chooise.
""")
if st.button('Show Price Distribution'):
    price_graph = plt.figure(figsize=(10,6))
    sns.distplot(df['price'])
    st.pyplot(price_graph)


if st.button('Show count of bedrooms'):
    br_graph = plt.figure(figsize=(10,6))
    sns.countplot(df['bedrooms'])
    st.pyplot(br_graph)

if st.button('Price by SQFT'):
    sq_graph = plt.figure(figsize=(10,6))
    sns.scatterplot(x='price',y='sqft_living',data=df)
    st.pyplot(sq_graph)

if st.button('Price by Bedrooms'):
    bed_graph = plt.figure(figsize=(10,6))
    sns.boxplot(x='bedrooms',y='price',data=df)
    st.pyplot(bed_graph)


non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

if st.button('By Region'):
    map_graph = plt.figure(figsize=(10,6))
    sns.scatterplot(x='long',y='lat',data=non_top_1_perc,
    hue='price', edgecolor=None,alpha=0.25,palette='RdYlGn')
    st.pyplot(map_graph)
