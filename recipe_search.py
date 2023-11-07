import streamlit as st
from streamlit_gsheets import GSheetsConnection
from sentence_transformers import SentenceTransformer

spreadsheet = "https://docs.google.com/spreadsheets/d/1oboE6E4MVZ538GilwB2Pvgm6249QxT6E8lkJzsL4yEk/edit#gid=0"

st.set_page_config(
    page_title="Recipe Search Engine",
    page_icon=":female-cook:"
)

st.title("Recipe Search Engine :shallow_pan_of_food: :female-cook:")

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read()

df = df.iloc[:100]

st.text_input("Search for recipes")

st.write(df.head())

sentences = list(df["title"])

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)

st.write(embeddings)

