import streamlit as st
from streamlit_gsheets import GSheetsConnection
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

@st.cache_data(ttl=3600*24*30)
def create_search_embedding(sentence: str) -> torch.Tensor:
    """
    """
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    search_embedding = model.encode(sentence)
    return search_embedding

@st.cache_data(ttl=3600*24*30)
def find_search_results(embeddings: torch.Tensor, search_embedding: torch.Tensor) -> pd.DataFrame:
    """
    
    """
    results = []
    search_emb = search_embedding.reshape(1,-1)
    for n_emb, embedding in enumerate(embeddings):
        row = {}
        emb = embedding.reshape(1,-1)

        cos_sim = cosine_similarity(search_emb, emb)[0]
        row["index"] = n_emb
        row["cos_sim"] = cos_sim
        results.append(row)
        
    return pd.DataFrame(results).sort_values(by="cos_sim", ascending=False)

st.title("Recipe Search Engine :shallow_pan_of_food: :female-cook:")

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read()

# just play around with 100 examples for now
df = df.iloc[:1000]

text_search = st.text_input("Search for your favorite recipes!")

embeddings = presearch(df)

# st.write(df.head())

# st.write(embeddings)

if text_search:
    search_embedding = create_search_embedding(text_search)
    df_search = find_search_results(embeddings, search_embedding)

    df_search = df_search.merge(df, left_index=True, right_index=True)

    for n_row, row in df_search.reset_index().iterrows():
        # draw a line if there is a new row
        if n_row%CARDS_PER_ROW == 0:
            st.write("---")
            cols = st.columns(CARDS_PER_ROW, gap="large")
        with cols[n_row%CARDS_PER_ROW]:
            st.write(f"{row['title']}")
            st.markdown("Ingredients: " + row['NER'][1:-1])
            # for item in row['NER'][1:-1].split(","):
            #     st.markdown(" - " + item.strip('"').strip())
            st.write(f"{row['link']}")
            st.write(f"Score: {row['cos_sim'][0]:.2f}")

