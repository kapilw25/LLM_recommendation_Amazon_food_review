# run Streamlit using the Python executable explicitly:
# conda activate rag_rec
# python -m streamlit run app_5_streamlit.py



# conda activate rag_rec
import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit app
st.title("Amazon Shopping Recommendation")

# Load the documents only once using Streamlit's caching
# @st.cache(allow_output_mutation=True)
# st.cache is deprecated. Please use one of Streamlit's new caching commands, st.cache_data or st.cache_resource.
@st.cache_resource
def load_data():
        # Load the documents
        loader = CSVLoader(file_path="Grocery_and_Gourmet_Food_filtered_1000.csv") #make it 20k

        # build embeddings model via OpenAI API
        embeddings_model = OpenAIEmbeddings(api_key=api_key) # replace openAI emebeddings with local [FaiSS, Ollama: nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed] embeddings which will train on GPU

        # Create an index using the loaded documents with the correct embeddings model
        index_creator = VectorstoreIndexCreator(embedding=embeddings_model)
        docsearch = index_creator.from_loaders([loader])
        
        return docsearch

docsearch = load_data()

# Create a question-answering chain using the index
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=api_key),\
    chain_type="stuff",\
    retriever=docsearch.vectorstore.as_retriever(),\
    input_key="question"
)

# User input
input_dish = st.text_input("Enter the dish you want to cook:",\
        placeholder="Enter a dish here...")


user_query = f"from the column named 'title' in the dataset,\
                recommend me 5 items to cook [not ready made] {input_dish}.\
                Do not recommend similar items of the same category.\
                e.g., do not recommend 2 types of rice as the same ingredient."

# When the user presses the 'Recommend Shopping List' button
if st.button("Recommend Shopping List"):
    response = chain({"question": user_query})
    if 'result' in response:
        st.write("Recommended Shopping List:")
        st.write(response['result'])
    else:
        st.write("No recommendations found")



