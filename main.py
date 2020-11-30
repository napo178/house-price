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

if st.button('Show the code'):
    code = """model = Sequential()

model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')"""
    st.code(code,language='python')

st.write("""
Here you can see some usefull charts by your chooise.
""")

selector = st.selectbox('Chose the graph',('Price Distribution','Number of bedrooms','Price by SQFT','Price by bedrooms','Price by region','Price by waterfront'))
if selector == 'Price Distribution':
    price_graph = plt.figure(figsize=(10,6))
    sns.distplot(df['price'],color='crimson')
    st.pyplot(price_graph)

if selector == 'Number of bedrooms':
    br_graph = plt.figure(figsize=(10,6))
    sns.countplot(df['bedrooms'])
    st.pyplot(br_graph)

if selector == 'Price by bedrooms':
    bed_graph = plt.figure(figsize=(10,6))
    sns.boxplot(x='bedrooms',y='price',data=df)
    st.pyplot(bed_graph)

if selector == 'Price by SQFT':
    sq_graph = plt.figure(figsize=(10,6))
    sns.scatterplot(x='price',y='sqft_living',data=df,color='crimson')
    st.pyplot(sq_graph)

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

if selector == 'Price by waterfront':
    wt_graph = plt.figure(figsize=(10,6))
    sns.boxplot(x='waterfront',y='price',data=df)
    st.pyplot(wt_graph)

if selector == 'Price by region':
    map_graph = plt.figure(figsize=(10,6))
    sns.scatterplot(x='long',y='lat',data=non_top_1_perc,
    hue='price', edgecolor=None,alpha=0.25,palette='RdYlGn')
    st.pyplot(map_graph)
