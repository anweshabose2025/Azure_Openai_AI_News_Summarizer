# AZURE OPENAI LLM
# streamlit run 2.py

import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from datetime import datetime

st.set_page_config(page_title="Agentic AI News Summarizer", layout="wide")
st.sidebar.header("Azure Openai")
st.sidebar.markdown("[Create Azure Openai Resource](portal.azure.com)")
AZURE_OPENAI_API_KEY = st.sidebar.text_input("Openai API Key of Azure resourse", type="password")
AZURE_ENDPOINT_URI = st.sidebar.text_input("Openai Endpoint URI of Azure resourse")
API_VERSION = st.sidebar.text_input("Openai API version")
DEPLOYMENT = st.sidebar.text_input("Openai Deployment name")

TAVILY_API_KEY = st.sidebar.text_input("Tavily API Key", type="password")
st.sidebar.markdown("[Get free TAVILY API Key](https://tavily.com)")

# UI Body
st.title("📰 Agentic AI News Summarizer with Azure OpenAI")
st.write("👋 Hi! Want the latest AI digest? You're in the right place.")

st.subheader("🗞️ AI News Explorer")
st.write("Please select the timeframe of AI news you want to explore.")

timeframe = st.radio("Select timeframe", ["Daily", "Weekly", "Monthly"], index=1)
st.write(f"You selected: **{timeframe}**")


class NewsState(TypedDict):
    timeframe: str
    raw_news: List[str]
    summary: str


def fetch_news(state: NewsState):
    """Fetch AI news based on the specified frequency.
    Args:
        state (dict): The state dictionary containing 'frequency'.
    Returns:
        dict: Updated state with 'news_data' key containing fetched news."""
    timeframe = state["timeframe"]
    time_range_map = {'Daily': 'd', 'Weekly': 'w', 'Monthly': 'm', 'Yearly': 'y'}
    days_map = {'Daily': 1, 'Weekly': 7, 'Monthly': 30, 'Yearly': 365}

    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily.search(
        api_key = TAVILY_API_KEY,
        query="Top Artificial Intelligence (AI) technology news India and globally",
        topic="news",
        time_range=time_range_map[timeframe],
        include_answer="advanced",
        max_results=20,
        days=days_map[timeframe])
    return {"raw_news":response.get('results', [])}


def summarize_news(state: NewsState):
    model = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT_URI,
        azure_deployment=DEPLOYMENT,
        api_version=API_VERSION)

    prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Summarize AI news articles into markdown format. For each item include:
            - Date in **YYYY-MM-DD** format in IST timezone
            - Concise sentences summary from latest news
            - Sort news by date wise (latest first)
            - Source URL as link
            Use format:
            ### [Date]
            - [Summary](URL)"""),
            ("user", "Articles:\n{articles}")])
    news_items = state["raw_news"]
    articles_str = "\n\n".join([f"Content: {item.get('content', '')}\nURL: {item.get('url', '')}\nDate: {item.get('published_date', '')}" for item in news_items])
    response = model.invoke(prompt_template.format(articles=articles_str))

    state["summary"] = response.content
    return state


workflow = StateGraph(NewsState)
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("summarize_news", summarize_news)
workflow.add_edge(START, "fetch_news")
workflow.add_edge("fetch_news", "summarize_news")
workflow.add_edge("summarize_news", END)
app = workflow.compile()

if timeframe and st.button("Get the Newzz"):
    with st.spinner("Fetching and summarizing news... ⏳"):
        final_state = app.invoke({"timeframe": timeframe, "raw_news": [], "summary": ""})

        st.subheader("🔍 Summary")
        summary_text = final_state["summary"]
        st.markdown(summary_text)

    # Download the News
    filename = f"AI_News_{timeframe}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.md"
    st.download_button(label="Download these Newz as Markdown",data=summary_text,file_name=filename,mime="text/markdown", icon=":material/download:")
    st.success("✅ Click the above button to download the File !! 😃")