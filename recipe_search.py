import streamlit as st
from streamlit_gsheets import GSheetsConnection

spreadsheet = "https://docs.google.com/spreadsheets/d/1oboE6E4MVZ538GilwB2Pvgm6249QxT6E8lkJzsL4yEk/edit#gid=0"

st.title("Recipe Search")

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read()

st.write(df.head())

