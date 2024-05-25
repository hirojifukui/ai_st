import streamlit as st
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0, 100, (20,5)), columns=("LA", "Math", "Physics", "SS", "PE"))

#Table 
st.sidebar.title("Test Results")
st.sidebar.dataframe(df) 

#Bar Graph
# st.title("Math")
# st.bar_chart(df["LA", "Math"])
# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

# st.bar_chart(chart_data)
# #Line Graph
# st.title("Physics")
# st.line_chart(df["Physics"])

# #Scatter plot
# df["Total"]=df["LA"]+df["Math"]+df["Physics"]+df["SS"]+df["PE"]
# st.title("Math & Physics")
# st.scatter_chart(df, x = "Physics", y = "Math", size="Total")

#Map with Scatter plot
# st.title("San Francisco")
df = pd.DataFrame(
    np.random.randn(200, 2) / [50, 50] + [37.76, -122.45],
    columns=['lat', 'lon'])

st.map(df)