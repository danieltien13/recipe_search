import streamlit as st
from streamlit_gsheets import GSheetsConnection
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

spreadsheet = "https://docs.google.com/spreadsheets/d/1oboE6E4MVZ538GilwB2Pvgm6249QxT6E8lkJzsL4yEk/edit#gid=0"

# constants
CARDS_PER_ROW = 3

st.set_page_config(
    page_title="Recipe Search Engine",
    page_icon=":female-cook:"
)

@st.cache_data(ttl=3600*24*30)
def presearch(df: pd.DataFrame) -> torch.Tensor:
    """
    Run all necessary computations before the search starts including:
    1. Loading the NLP sentence similarity model
    2. Generating embeddings (ideally these are pre-generated and stored in BigQuery)

    Returns the embeddings that captures sentence semantics 
    """
    sentences = list(df["title"])
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    return embeddings

st.title("Recipe Search Engine :shallow_pan_of_food: :female-cook:")

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read()

# just play around with 100 examples for now
df = df.iloc[:100]

text_search = st.text_input("Search for recipes")

embeddings = presearch(df)

st.write(df.head())

# st.write(embeddings)

df_search = df

if text_search:
    for n_row, row in df_search.reset_index().iterrows():
        # draw a line if there is a new row
        if n_row%CARDS_PER_ROW == 0:
            st.write("---")
            cols = st.columns(CARDS_PER_ROW, gap="large")
        with cols[n_row%CARDS_PER_ROW]:
            st.write(f"{row['title']}")
            st.write(f"{row['NER']}")
            st.write(f"{row['link']}")

