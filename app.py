import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from helper import HelperClass

# load the models
model = pickle.load(open('model_dtr.pkl','rb'))



# ===========================================================
# python main.........................................
if __name__ == "__main__":
    # Predictive System
    st.title("Countries GDP Analysis Dashboard and Prediction System")
    features = st.text_input("Input features")
    if st.button("Predict GDP"):
        features = features.split(',')
        features = np.array([features])
        gdp_pred = model.predict(features).reshape(1,-1)
        st.write("Predict GDP Per Capita :", gdp_pred[0])




    # side bar code
    st.sidebar.title("Data Uploader")
    file = st.sidebar.file_uploader('Upload CSV File', type=['csv'])
    if file is not None:
        # reading data
        df = pd.read_csv(file)
        # calling basic_counts fun
        region, countries, countries_counts = HelperClass.basic_counts(df)
        st.sidebar.write("Total Region :", region)
        st.sidebar.write("Total Countries :", countries)
        st.sidebar.write("Total Countries Per Each Region :", countries_counts)

        if st.sidebar.button('Show Analysis Dashboard'):
            df = HelperClass.ConvertToFloatAndFillMissValues(df)
            st.subheader("Data View")
            st.write(df)

            col1, col2 = st.columns(2)
            with col1:
                # # Calculate the median for the specified columns by region
                st.subheader("Average GDP, Literature, Agriculture Per Region")
                result = HelperClass.AverageRegionsGDPLiteracyAgriculture(df)
                st.write(result)
            with col2:
                # data agg
                data_agg = HelperClass.DataAgg(df)
                print(data_agg, "kjljklj")
                st.subheader("Data Aggregation Per region")
                st.write(data_agg)

            # Top 15 Countries GDP per capita
            st.subheader("Top 15 Countries GDP Per Capita")
            HelperClass.plot_gdp_bar_chart(df)
            # Top 5 Asian Countries GDP,Literature
            st.subheader("Top 5 Asian Countries GDP, Literature")
            HelperClass.AsiaFiveRegionGDP(df)

            # Top five countries GDP per each Region
            st.subheader("Top Five Countries GDP Per Each Region")
            HelperClass.EachReginGDP(df)
