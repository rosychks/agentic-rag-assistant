App · PY
import os
import io
import time
import base64
 
import fitz
import gradio as gr
from PIL import Image
from openai import RateLimitError
from pydantic import BaseModel, Field
from typing import List, TypedDict
 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langgraph.graph import StateGraph, END
 
# ---------------------------------------------------------------------------
# 0. Ключи (берутся из переменных окружения, заданных в настройках Render)
# ---------------------------------------------------------------------------
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Не задан OPENAI_API_KEY в переменных окружения")
# TAVILY_API_KEY опционален — без него web_search просто вернёт заглушку
 
PDF_PATH = os.path.join(os.path.dirname(__file__), "test_document.pdf")

---------------------------------------------------------------------------
# 1. Ingest (текст + мультимодальные описания картинок)
# ---------------------------------------------------------------------------
vision_llm = ChatOpenAI(model="gpt-4o-mini")
 
 
def caption_image(png_bytes, retries=5):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
 
    for attempt in range(retries):
        try:
            msg = vision_llm.invoke([
                {"role": "user", "content": [
                    {"type": "text", "text": "Опиши, что изображено на картинке, кратко и по делу."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ])
            return msg.content
        except RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"Rate limit, жду {wait} сек...")
            time.sleep(wait)
    return "(не удалось получить описание — лимит токенов)"
 
 
def ingest_pdf(path, max_pages=8):
    docs = []
    pdf = fitz.open(path)
    n_pages = min(max_pages, pdf.page_count)
    for page in pdf[:n_pages]:
        text = page.get_text()
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path, "page": page.number}))
        for img in page.get_images():
            xref = img[0]
            base_image = pdf.extract_image(xref)
            caption = caption_image(base_image["image"])
            docs.append(Document(page_content=f"[Image] {caption}", metadata={"source": path, "page": page.number}))
            time.sleep(1)
    return docs
 
 
# ---------------------------------------------------------------------------
# 2. Чанкинг + эмбеддинги + Qdrant (локальный диск — переживает "сон" сервиса,
#    но обнуляется при новом деплое)
# ---------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
 
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_data")
qdrant_client = QdrantClient(path=QDRANT_PATH)
 
collection_exists = qdrant_client.collection_exists("docs")
 
if not collection_exists:
    print("Коллекция не найдена — индексирую PDF заново (это займёт время)...")
    all_docs = ingest_pdf(PDF_PATH, max_pages=46)
    print("Собрано документов (текст+картинки):", len(all_docs))
 
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(all_docs)
    print("Чанков получилось:", len(chunks))
 
    qdrant_client.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(client=qdrant_client, collection_name="docs", embedding=embeddings)
    vectorstore.add_documents(chunks)
    print("Индексация завершена и сохранена на диск")
else:
    print("Найдена готовая коллекция на диске — пропускаю индексацию")
    vectorstore = QdrantVectorStore(client=qdrant_client, collection_name="docs", embedding=embeddings)
 
retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
print("Векторная база готова")
 
# ---------------------------------------------------------------------------
# 3. LangGraph агент
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
 
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    steps: List[str]
    web_fallback_used: bool
    retries: int
 
 
class YesNo(BaseModel):
    binary_score: str = Field(description="'yes' или 'no'")
 
 
grade_prompt = ChatPromptTemplate.from_template(
    "Документ:\n{document}\n\nВопрос: {question}\n\n"
    "Релевантен ли документ вопросу? Ответь 'yes' или 'no'."
)
grader = grade_prompt | llm.with_structured_output(YesNo)
 
 
def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "steps": state.get("steps", []) + ["retrieve"]}
 
 
def grade_documents(state):
    good_docs = []
    for d in state["documents"]:
        score = grader.invoke({"document": d.page_content, "question": state["question"]})
        if score.binary_score == "yes":
            good_docs.append(d)
        time.sleep(1)
    return {"documents": good_docs, "steps": state["steps"] + ["grade_documents"]}
 
 
def route_after_grade(state):
    return "web_search" if len(state["documents"]) == 0 else "generate"
 
 
def web_search(state):
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        results = TavilySearchResults(k=3).invoke(state["question"])
        web_docs = [Document(page_content=r["content"]) for r in results]
    except Exception:
        web_docs = [Document(page_content="(веб-поиск недоступен, ключ Tavily не задан)")]
    return {
        "documents": state["documents"] + web_docs,
        "web_fallback_used": True,
        "steps": state["steps"] + ["web_search"],
    }
 
 
gen_prompt = ChatPromptTemplate.from_template(
    "Ты должен отвечать СТРОГО на основе контекста ниже, без использования своих общих знаний.\n"
    "Если контекст не содержит прямого ответа на вопрос — так и скажи: "
    "'Не могу ответить на этот вопрос — в документах и веб-поиске не нашлось релевантной информации.' "
    "Не додумывай факты и не используй информацию, которой нет в контексте, даже если ты её знаешь.\n\n"
    "Контекст:\n{context}\n\nВопрос: {question}"
)
 
 
def generate(state):
    context_docs = state["documents"]
    is_empty = len(context_docs) == 0 or all(
        "недоступен" in d.page_content or len(d.page_content.strip()) < 15
        for d in context_docs
    )
    if is_empty:
        return {
            "generation": "Не могу ответить на этот вопрос — в документах и веб-поиске не нашлось релевантной информации.",
            "steps": state["steps"] + ["generate"],
            "retries": state.get("retries", 0) + 1,
        }
    context = "\n\n".join(f"[стр. {d.metadata.get('page', '?')}] {d.page_content}" for d in context_docs)
    answer = (gen_prompt | llm).invoke({"context": context, "question": state["question"]})
    return {
        "generation": answer.content,
        "steps": state["steps"] + ["generate"],
        "retries": state.get("retries", 0) + 1,
    }
 
 
hallucination_prompt = ChatPromptTemplate.from_template(
    "Контекст:\n{context}\n\nОтвет:\n{generation}\n\n"
    "Обоснован ли ответ контекстом (нет придуманных фактов)? 'yes' или 'no'."
)
hallucination_grader = hallucination_prompt | llm.with_structured_output(YesNo)
 
answer_prompt = ChatPromptTemplate.from_template(
    "Вопрос: {question}\n\nОтвет: {generation}\n\n"
    "Отвечает ли этот ответ на вопрос? 'yes' или 'no'."
)
answer_grader = answer_prompt | llm.with_structured_output(YesNo)
 
 
def route_after_generate(state):
    if "Не могу ответить" in state["generation"]:
        return "useful"
    if state["retries"] >= 3:
        return "useful"
    context = "\n\n".join(d.page_content for d in state["documents"])
    grounded = hallucination_grader.invoke({"context": context, "generation": state["generation"]})
    if grounded.binary_score == "no":
        return "not_grounded"
    useful = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
    return "useful" if useful.binary_score == "yes" else "not_useful"
 
 
g = StateGraph(GraphState)
g.add_node("retrieve", retrieve)
g.add_node("grade_documents", grade_documents)
g.add_node("web_search", web_search)
g.add_node("generate", generate)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "grade_documents")
g.add_conditional_edges("grade_documents", route_after_grade, {"web_search": "web_search", "generate": "generate"})
g.add_edge("web_search", "generate")
g.add_conditional_edges("generate", route_after_generate,
                         {"useful": END, "not_grounded": "generate", "not_useful": "web_search"})
graph_app = g.compile()
print("Граф собран")
 
# ---------------------------------------------------------------------------
# 4. Retry-обёртка (защита от rate limit)
# ---------------------------------------------------------------------------
 
 
def invoke_with_retry(question, retries=6):
    for attempt in range(retries):
        try:
            return graph_app.invoke({"question": question, "steps": [], "retries": 0, "web_fallback_used": False})
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"Rate limit, жду {wait} сек...")
            time.sleep(wait)
    raise RuntimeError("Не удалось выполнить после нескольких попыток")
 
 
# ---------------------------------------------------------------------------
# 5. Gradio-интерфейс
# ---------------------------------------------------------------------------
 
 
def chat_fn(question, history):
    r = invoke_with_retry(question)
    steps_str = " → ".join(r["steps"])
    return f"{r['generation']}\n\n*шаги: {steps_str}*"
 
 
demo = gr.ChatInterface(
    chat_fn,
    title="Agentic RAG Assistant",
    description=(
        "Задайте вопрос по книге NASA/Stanford Solar Center «Our Solar System — "
        "Ancient Worlds, New Discoveries»: о Солнце, планетах, их спутниках, "
        "астероидах, кометах и явлениях космической погоды. "
        "Если ответа нет в документе, бот честно скажет, что не знает "
        "— он не выдумывает факты."
    ),
    examples=[
        "Что такое солнечные пятна и как часто они появляются?",
        "Что такое корона и солнечный ветер?",
        "Расскажи про кольца Сатурна",
        "Сколько спутников у Марса?",
        "Что показал спутник SOHO?",
    ],
)
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
