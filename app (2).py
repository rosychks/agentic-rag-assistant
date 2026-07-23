import os
import io
import re
import time
import base64

import fitz
import gradio as gr
from PIL import Image
from openai import RateLimitError
from typing import List, TypedDict, Any
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# 0. Ключи — используется только DeepSeek, эмбеддинги не нужны вообще
#    (поиск по документу идёт через BM25, это не нейросеть, а статистический
#    алгоритм текстового поиска, работает без каких-либо ключей).
# ---------------------------------------------------------------------------
if not os.environ.get("DEEPSEEK_API_KEY"):
    raise RuntimeError("Не задан DEEPSEEK_API_KEY в переменных окружения")

# ---------------------------------------------------------------------------
# 1. Ingest (текст + опционально мультимодальные описания картинок)
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)
# Мультимодальность (описание картинок) у DeepSeek нестабильна/бета — используем
# тот же llm, но по умолчанию галочка "анализировать картинки" выключена в интерфейсе.
vision_llm = llm


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
            time.sleep(wait)
    return "(не удалось получить описание — лимит токенов)"


def ingest_pdf(path, max_pages=60, include_images=False):
    docs = []
    pdf = fitz.open(path)
    n_pages = min(max_pages, pdf.page_count)
    for page in pdf[:n_pages]:
        text = page.get_text()
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path, "page": page.number}))
        if include_images:
            for img in page.get_images():
                xref = img[0]
                base_image = pdf.extract_image(xref)
                caption = caption_image(base_image["image"])
                docs.append(Document(page_content=f"[Image] {caption}", metadata={"source": path, "page": page.number}))
                time.sleep(1)
    return docs, pdf.page_count


# ---------------------------------------------------------------------------
# 2. Построение индекса из загруженного пользователем файла (по требованию,
#    не при старте сервера — поэтому порт открывается мгновенно)
# ---------------------------------------------------------------------------


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Простой ретривер на BM25 — без эмбеддингов и без внешних ключей."""

    def __init__(self, documents):
        self.documents = documents
        tokenized_corpus = [tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def invoke(self, query, k=12):
        scores = self.bm25.get_scores(tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked_idx[:k] if scores[i] > 0]


def process_upload(file, include_images):
    if file is None:
        return "Сначала выберите PDF-файл.", None

    try:
        all_docs, total_pages = ingest_pdf(file.name, max_pages=60, include_images=include_images)
        if not all_docs:
            return "В этом PDF не нашлось текста для индексации.", None

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(all_docs)
        retriever = BM25Retriever(chunks)

        status = (
            f"✅ Готово! Обработано страниц: {min(60, total_pages)} из {total_pages}, "
            f"чанков: {len(chunks)}. Можно задавать вопросы."
        )
        return status, retriever
    except Exception as e:
        return f"❌ Ошибка при обработке файла: {e}", None


# ---------------------------------------------------------------------------
# 3. LangGraph агент (retriever приходит через state — свой для каждой сессии,
#    а не общий на все соединения, чтобы разные пользователи не путали документы)
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    question: str
    retriever: Any
    documents: List[Document]
    generation: str
    steps: List[str]
    web_fallback_used: bool
    retries: int


grade_prompt = ChatPromptTemplate.from_template(
    "Документ:\n{document}\n\nВопрос: {question}\n\n"
    "Оцени, содержит ли документ ХОТЯ БЫ ЧАСТИЧНО полезную информацию по теме вопроса "
    "(не обязательно исчерпывающий ответ — достаточно, чтобы документ был по той же теме "
    "или мог помочь составить ответ). Будь снисходителен: если есть разумные сомнения "
    "в релевантности, отвечай 'yes'. Отвечай 'no' только если документ совершенно не по теме. "
    "Ответь ОДНИМ словом: yes или no, без пояснений."
)
grader_chain = grade_prompt | llm


def grade_one(document_text, question):
    response = grader_chain.invoke({"document": document_text, "question": question})
    return response.content.strip().lower().startswith("y")


def retrieve(state):
    retriever = state["retriever"]
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "steps": state.get("steps", []) + ["retrieve"]}


def grade_documents(state):
    good_docs = []
    for d in state["documents"]:
        is_relevant = grade_one(d.page_content, state["question"])
        if is_relevant:
            good_docs.append(d)
        time.sleep(0.5)
    if len(good_docs) == 0 and len(state["documents"]) > 0:
        good_docs = state["documents"][:3]
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
    "Ты отвечаешь на основе контекста ниже. Контекст может состоять из разрозненных фрагментов "
    "текста и подписей к картинкам — это нормально, СОБЕРИ ответ из всех подходящих фрагментов, "
    "даже если ни один из них по отдельности не даёт полного ответа. "
    "Используй только факты, присутствующие в контексте (не добавляй сведения из общих знаний), "
    "но если в контексте есть хоть какая-то релевантная информация по теме вопроса — "
    "сформулируй из неё связный ответ, а не отказывайся.\n"
    "Только если контекст ВООБЩЕ не связан с темой вопроса, ответь: "
    "'Не могу ответить на этот вопрос — в документе и веб-поиске не нашлось релевантной информации.'\n\n"
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
            "generation": "Не могу ответить на этот вопрос — в документе и веб-поиске не нашлось релевантной информации.",
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
    "Обоснован ли ответ контекстом (нет придуманных фактов)? "
    "Ответь ОДНИМ словом: yes или no, без пояснений."
)
hallucination_chain = hallucination_prompt | llm

answer_prompt = ChatPromptTemplate.from_template(
    "Вопрос: {question}\n\nОтвет: {generation}\n\n"
    "Отвечает ли этот ответ на вопрос? Ответь ОДНИМ словом: yes или no, без пояснений."
)
answer_chain = answer_prompt | llm


def route_after_generate(state):
    if "Не могу ответить" in state["generation"]:
        return "useful"
    if state["retries"] >= 3:
        return "useful"
    context = "\n\n".join(d.page_content for d in state["documents"])
    grounded_response = hallucination_chain.invoke({"context": context, "generation": state["generation"]})
    is_grounded = grounded_response.content.strip().lower().startswith("y")
    if not is_grounded:
        return "not_grounded"
    useful_response = answer_chain.invoke({"question": state["question"], "generation": state["generation"]})
    is_useful = useful_response.content.strip().lower().startswith("y")
    return "useful" if is_useful else "not_useful"


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


def invoke_with_retry(question, retriever, retries=6):
    for attempt in range(retries):
        try:
            return graph_app.invoke({
                "question": question,
                "retriever": retriever,
                "steps": [],
                "retries": 0,
                "web_fallback_used": False,
            })
        except RateLimitError:
            wait = 20 * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError("Не удалось выполнить после нескольких попыток")


# ---------------------------------------------------------------------------
# 4. Интерфейс: загрузка PDF + чат
# ---------------------------------------------------------------------------

with gr.Blocks(title="Agentic RAG Assistant") as demo:
    gr.Markdown("# Agentic RAG Assistant")
    gr.Markdown(
        "Загрузите свой PDF-файл, дождитесь обработки — и задавайте вопросы по его содержимому. "
        "Можно в любой момент загрузить другой файл вместо текущего."
    )

    with gr.Row():
        file_input = gr.File(label="PDF-документ", file_types=[".pdf"])
        include_images = gr.Checkbox(
            label="Анализировать картинки в PDF (точнее, но заметно медленнее)",
            value=False,
        )

    process_btn = gr.Button("Обработать документ", variant="primary")
    status = gr.Markdown("Документ ещё не загружен.")
    retriever_state = gr.State(None)

    chatbot = gr.Chatbot(label="Чат-бот", height=400)
    question_box = gr.Textbox(label="Ваш вопрос", placeholder="Сначала загрузите и обработайте PDF...")
    clear_btn = gr.Button("Очистить чат")

    process_btn.click(
        fn=process_upload,
        inputs=[file_input, include_images],
        outputs=[status, retriever_state],
    )

    def respond(question, history, retriever):
        if not question or not question.strip():
            return history, ""
        if retriever is None:
            history = history + [[question, "Сначала загрузите PDF-файл и нажмите «Обработать документ»."]]
            return history, ""
        r = invoke_with_retry(question, retriever)
        steps_str = " → ".join(r["steps"])
        answer = f"{r['generation']}\n\n*шаги: {steps_str}*"
        history = history + [[question, answer]]
        return history, ""

    question_box.submit(respond, inputs=[question_box, chatbot, retriever_state], outputs=[chatbot, question_box])
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
