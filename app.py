import streamlit as st

st.set_page_config(layout="wide")

from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from ask_llm import get_llm_answer, OpenAI

from dotenv import load_dotenv


load_dotenv()
COLLECTION_NAME = "LuXunWorks"
MILVUS_ENDPOINT = "./milvus.db"


# Logo
st.image("./pics/Milvus_Logo_Official.png", width=200)

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">RAG of LuXun's works</div>
    <div class="description">
        This chatbot is built with Milvus vector database, supported by OpenAI text embedding model.<br>
        It supports conversation based on knowledge from the Milvus development guide document.
    </div>
    """,
    unsafe_allow_html=True,
)

# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
openai_client = OpenAI()

retrieved_lines_with_distances = []

with st.form("my_form"):
    question = st.text_area("Enter your question:")
    # Sample question: what is the hardware requirements specification if I want to build Milvus and run from source code?
    submitted = st.form_submit_button("Submit")

    if question and submitted:
        # generator = get_generator()

        # search_res = generator.search(question)
        query_vector = emb_text(openai_client, question)
        search_res = get_search_results(milvus_client=milvus_client, collection_name=COLLECTION_NAME, query_vector=query_vector, output_fields=["*"])[0]
        context = [{"title": res["entity"]["title"], "type": res["entity"]["type"], "date": res["entity"]["date"], "window": res["entity"]["window"]} for res in search_res]
        retrieved_lines_with_distances = [
            (res["entity"]["window"], res ["distance"]) for res in search_res
        ]

        # Create context from retrieved lines
        answer = get_llm_answer(openai_client, str(context), question)

        # Display the question and response in a chatbot-style box
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)


# Display the retrieved lines in a more readable format
    st.sidebar.subheader("Retrieved Lines with Distances:")
    for idx, (line, distance) in enumerate(retrieved_lines_with_distances, 1):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Result {idx}:**")
        st.sidebar.markdown(f"> {line}")
        st.sidebar.markdown(f"*Distance: {distance:.2f}*")
