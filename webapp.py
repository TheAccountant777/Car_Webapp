import numpy as np
import streamlit as st
import pandas as pd
from pycaret.regression import *

model = load_model('Model_file')
data = pd.read_csv('car_data.csv')

# ------------------------------------ Design --------------------------#
st.image("image.jpg")
st.title("Used car price predictor!")
st.write("""
### This program will give you an estimate of how much a car is worth in the kenyan market.
""")
st.write("  ")

st.write('### Select the car properties.')


# ------------------------ Create Data Frame ----------------------------#

brands = sorted(data.brand.unique())
car_brand = st.selectbox("What is the brand?",brands)
car_model = st.selectbox("What is the model?", sorted(data[data['brand'] == car_brand]['model'].unique()))
car_year = st.selectbox("What year was it made?", sorted(data[data['model'] == car_model]['year'].unique()))
car_trans = st.selectbox("Transmition", sorted(['Automatic', 'Manual']))
car_fuel = st.selectbox("Fuel", sorted(['Petrol', 'Diesel']))
car_millage = st.select_slider('Select the millage', range(10000,200001))
car_size = st.selectbox('What is the engine size? ', sorted(data[data['model'] == car_model]['disp.'].unique()))
car_age = 2021 - int(car_year)
km_year = car_millage / car_age

df_data = {'brand':[car_brand],
        'model': [car_model],
        'age': [car_age],
        'millage': [car_millage],
        'disp.' : [car_size],
        'fuel' :[car_fuel],
        'trans.': [car_trans],
        'km_year':[km_year]}

predict_data = pd.DataFrame(df_data)

# ------------------------ Predict ------------------#
def short(x):
  if len(str(int(x))) == 7:
    return str(int(x))[0] + "." + str(int(x))[1:3] + "M"
  else:
      return str(int(x))[0:3] +  "K"

st.write("")
if st.button('Get Price'):
    price = predict_model(model, predict_data)['Label'][0]
    price_upper = (price * 0.05) + price
    price_lower = price - (price * 0.05)
    st.write(f'''
    ## The price is between Ksh: {short(price_lower)} and {short(price_upper)}
    ''')




