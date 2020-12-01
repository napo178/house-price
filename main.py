import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


st.set_page_config(page_title='Predict House Price', page_icon='house')

df = pd.read_csv("house_data.csv")

st.write("""
### Predicting housing price in King County.
""")
st.write("""
This data is taken from Kaggle: [[DATA](https://www.kaggle.com/harlfoxem/housesalesprediction)]
""")

st.write("""
In this project we are going to use Neural Networks to predict the price of the house. It's a 5 layers Neural Network and you can see hows easy is to build one. It's only takess couple lines of code to do it and you can see the code by clicking on button below.
""")

if st.button('Show the code'):
    code = """from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

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
    sns.countplot(df['bedrooms'],palette='magma')
    st.pyplot(br_graph)

if selector == 'Price by bedrooms':
    bed_graph = plt.figure(figsize=(10,6))
    sns.boxplot(x='bedrooms',y='price',data=df,palette='magma')
    st.pyplot(bed_graph)

if selector == 'Price by SQFT':
    sq_graph = plt.figure(figsize=(10,6))
    sns.scatterplot(x='price',y='sqft_living',data=df,color='crimson')
    st.pyplot(sq_graph)

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

if selector == 'Price by waterfront':
    wt_graph = plt.figure(figsize=(10,6))
    sns.boxplot(x='waterfront',y='price',data=df,palette='magma')
    st.pyplot(wt_graph)

if selector == 'Price by region':
    map_graph = plt.figure(figsize=(10,6))
    sns.scatterplot(x='long',y='lat',data=non_top_1_perc,
    hue='price', edgecolor=None,alpha=0.25,palette='RdYlGn')
    st.pyplot(map_graph)

st.write("""
If you want to see how to create same plots you can use the checkbox below.
""")

code_dist = """plt.figure(figsize=(10,6))
sns.distplot(data,color='crimson')"""
code_c = """plt.figure(figsize=(10,6))
sns.countplot(data,palette='magma')"""
code_box = """plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms',y='price',data=data,palette='magma')"""
code_sc = """plt.figure(figsize=(10,6))
sns.scatterplot(x='price',y='sqft_living',data=data,color='crimson')"""


radio_b = st.radio('Plots',('Distplot','Countplot','Boxplot','Scatterplot'))
if radio_b == 'Distplot':
    st.code(code_dist,language='python')
elif radio_b == 'Countplot':
    st.code(code_c,language='python')
elif radio_b == 'Boxplot':
    st.code(code_box,language='python')
else:
    st.code(code_sc,language='python')


st.subheader('Parameters that you provided')
st.sidebar.header('Please provide your parameters to calculate price of your house')


def user_input_features():
    bedrooms = st.sidebar.number_input('Number of Bedrooms',min_value=1,max_value=10,value=3,step=1)
    bathrooms = st.sidebar.number_input('Number of Bathrooms',min_value=1.0,max_value=5.0,value=1.0,step=0.25)
    sqft_living = st.sidebar.number_input('Square footage of the apartments interior living space',min_value=370,max_value=13540,value=1215)
    sqft_lot = st.sidebar.number_input('Square footage of the land space',min_value=520,max_value=1651359,value=1511)
    floors = st.sidebar.number_input('Number of floors',min_value=1,max_value=5,value=1)
    waterfront_c = st.sidebar.selectbox('Whether the apartment was overlooking the waterfront or not',('Yes','No'))
    view = st.sidebar.number_input('An index from 0 to 4 of how good the view of the property was',0,4,1)
    condition = st.sidebar.number_input('An index from 1 to 5 on the condition of the apartment',1,5,1)
    grade = st.sidebar.number_input('An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design',1,13,7)
    sqft_above = st.sidebar.number_input('The square footage of the interior housing space that is above ground level',370,9410,3619)
    sqft_basement = st.sidebar.number_input('The square footage of the interior housing space that is below ground level', 0,4820,1521)
    yr_built = st.sidebar.number_input('The year the house was initially built',min_value=0,max_value=2020,value=1960)
    yr_renovated = st.sidebar.number_input('The year of the house’s last renovation',0,2020,2010)
    lat = st.sidebar.slider('Lattitude', 47.1559,47.7776,47.5432)
    long = st.sidebar.slider('Longitude', -122.51899999999999,-121.315,-121.163)
    sqft_living15 = st.sidebar.number_input('The square footage of interior housing living space for the nearest 15 neighbors',min_value=399,max_value=6210,value=2471)
    sqft_lot15 = st.sidebar.number_input('The square footage of the land lots of the nearest 15 neighbors',min_value=651,max_value=871200,value=63123)
    date = st.sidebar.date_input('Date of the home sale',value=datetime.date(2015,7,9))
    if waterfront_c == 'Yes':
        waterfront = 1
    else:
        waterfront = 0
    data = {'bedrooms':bedrooms,
            'bathrooms':bathrooms,
            'sqft_living':sqft_living,
            'sqft_lot':sqft_lot,
            'floors':floors,
            'waterfront':waterfront,
            'view':view,
            'condition':condition,
            'grade':grade,
            'sqft_above':sqft_above,
            'sqft_basement':sqft_basement,
            'yr_built':yr_built,
            'yr_renovated':yr_renovated,
            'lat':lat,
            'long':long,
            'sqft_living15':sqft_living15,
            'sqft_lot15':sqft_lot15,
            'date':date}
    features = pd.DataFrame(data,index=[0])
    return features

inp_df = user_input_features()

inp_df['date'] = pd.to_datetime(inp_df['date'])
inp_df['year'] = inp_df['date'].apply(lambda date: date.year)
inp_df['month'] = inp_df['date'].apply(lambda date: date.month)
inp_df['day'] = inp_df['date'].apply(lambda date: date.day)
inp_df = inp_df.drop('date',axis=1)
st.write(inp_df)


loaded_model = load_model('project/',custom_objects=None,compile=True)

predictions = loaded_model.predict(inp_df)
preds = predictions.tolist()
new_preds = preds[0][0]
new_preds = str(new_preds)
st.write(new_preds[:-6])
