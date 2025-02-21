import streamlit as st

st.set_page_config(layout="wide")

from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from ask_llm import get_llm_answer, OpenAI

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
    <div class="title">鲁迅作品的RAG</div>
    <div class="description">
        该聊天机器人使用 Milvus 向量数据库构建，由文本嵌入模型支持。<br>
        它支持基于鲁迅作品中的知识的对话。
    </div>
    """,
    unsafe_allow_html=True,
)

# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
with st.sidebar:
    st.markdown("[获取 Siliconflow API key](https://docs.siliconflow.cn/cn/userguide/quickstart)")
    siliconflow_api_key = st.text_input("**Siliconflow API Key**", key="siliconflow_api_key", type="password")
    base_url = st.text_input("**base url**", key="base_url", value="https://api.siliconflow.cn/v1", placeholder="https://api.siliconflow.cn/v1")
    model_name = st.text_input("**model name**", key="model_name", value="deepseek-ai/DeepSeek-V2.5", placeholder="deepseek-ai/DeepSeek-V2.5")


openai_client = OpenAI(api_key=siliconflow_api_key, base_url=base_url)

retrieved_lines_with_distances = []

with st.form("my_form"):
    question = st.text_area("输入你的问题:")
    # Sample question: what is the hardware requirements specification if I want to build Milvus and run from source code?
    submitted = st.form_submit_button("提交")
    if submitted and not siliconflow_api_key:
        st.warning("请先输入 Siliconflow API Key")
        st.stop()
    if question and submitted:
        # generator = get_generator()

        # search_res = generator.search(question)
        query_vector = emb_text(question)
        search_res = get_search_results(milvus_client=milvus_client, collection_name=COLLECTION_NAME, query_vector=query_vector, output_fields=["*"])[0]
        context = [{"title": res["entity"]["title"], "type": res["entity"]["type"], "date": res["entity"]["date"], "window": res["entity"]["window"]} for res in search_res]
        retrieved_lines_with_distances = [
            (res["entity"]["window"], res ["distance"]) for res in search_res
        ]

        # Create context from retrieved lines
        answer = get_llm_answer(openai_client, str(context), question, model=model_name)

        # Display the question and response in a chatbot-style box
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)


# Display the retrieved lines in a more readable format
    st.sidebar.markdown("---")
    st.sidebar.subheader("检索到的片段和距离:")
    for idx, (line, distance) in enumerate(retrieved_lines_with_distances, 1):
        st.sidebar.markdown(f"**文档片段 {idx}:**")
        st.sidebar.markdown(f"> {line}")
        st.sidebar.markdown(f"*距离: {distance:.2f}*")
        st.sidebar.markdown("---")
