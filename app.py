import os
import json
import requests
import gradio as gr
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------------------------------------------
# Wikipedia Loader
# -----------------------------------------------------------------------------

def load_wikipedia_text(topic):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": topic,
        "prop": "extracts",
        "explaintext": "",
    }
    headers = {"User-Agent": "RAG-Colab/1.0"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No Wikipedia content found for: {topic}")
    return page["extract"]


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------

def chunk_text(text, chunk_size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


# -----------------------------------------------------------------------------
# Embeddings + VectorStore
# -----------------------------------------------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embeddings)


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

def retrieve_context(vectorstore, question, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    sources = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)
    return context, sources


# -----------------------------------------------------------------------------
# Prompt Builder
# -----------------------------------------------------------------------------

def build_rag_prompt(context, question):
    prompt = (
        "You are a study assistant.\n\n"
        "Use ONLY the context below to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Return ONLY valid JSON in this format:\n\n"
        "{\n"
        '  "answer": "short answer (2-5 sentences)",\n'
        '  "supporting_quotes": ["quote from context", "quote from context"],\n'
        '  "confidence": "low | medium | high",\n'
        '  "missing_info": "what information was missing if any"\n'
        "}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
    )
    return prompt


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

def route_question(question, llm_call):
    router_prompt = (
        "You are deciding what to do next for a Wikipedia Q&A bot.\n\n"
        "If the question is unclear or vague, return: CLARIFY\n"
        "If the question is clear and specific, return: RETRIEVE\n\n"
        "Examples:\n"
        '"What is it?" -> CLARIFY\n'
        '"Tell me about it" -> CLARIFY\n'
        '"What is the atmosphere of Mars?" -> RETRIEVE\n'
        '"When was diabetes discovered?" -> RETRIEVE\n\n'
        f"Question: {question}\n\n"
        "Return ONLY one word: RETRIEVE or CLARIFY\n"
    )
    decision = llm_call(router_prompt).strip().upper()
    return "CLARIFY" if "CLARIFY" in decision else "RETRIEVE"


# -----------------------------------------------------------------------------
# Groq LLM
# -----------------------------------------------------------------------------

def call_groq(prompt):
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            '{"answer": "API error: ' + str(e) + '",'
            '"supporting_quotes": [], '
            '"confidence": "low", '
            '"missing_info": "Groq API call failed."}'
        )


# -----------------------------------------------------------------------------
# RAG Pipeline
# -----------------------------------------------------------------------------

def answer_question(vectorstore, question):
    decision = route_question(question, call_groq)

    if decision == "CLARIFY":
        return {
            "answer": "Please ask a more specific question.",
            "supporting_quotes": [],
            "confidence": "N/A",
            "missing_info": "Question unclear",
            "sources": []
        }

    context, sources = retrieve_context(vectorstore, question)
    prompt = build_rag_prompt(context, question)
    raw = call_groq(prompt)

    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        parsed = {
            "answer": raw,
            "supporting_quotes": [],
            "confidence": "unknown",
            "missing_info": "Invalid JSON response"
        }

    parsed["sources"] = sources
    return parsed


# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------

vectorstore = None
current_topic = None


# -----------------------------------------------------------------------------
# Gradio Functions
# -----------------------------------------------------------------------------

def load_topic(topic):
    global vectorstore, current_topic

    if not topic.strip():
        return "Please enter a topic."

    try:
        text = load_wikipedia_text(topic.strip())
        chunks = chunk_text(text)
        vectorstore = create_vectorstore(chunks)
        current_topic = topic.strip()
        return f"Loaded '{topic}' — {len(chunks)} chunks indexed. You can now ask questions!"
    except Exception as e:
        return f"Error: {str(e)}"


def chat_fn(message, history):
    history = history or []

    if vectorstore is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please load a Wikipedia topic first using the box above."})
        return history, ""

    result = answer_question(vectorstore, message)

    output = f"**Answer:** {result['answer']}\n\n"
    output += f"**Confidence:** {result['confidence']}\n\n"

    if result.get("supporting_quotes"):
        output += "**Supporting Quotes:**\n"
        for quote in result["supporting_quotes"]:
            output += f"> {quote}\n\n"

    if result.get("missing_info") and result["missing_info"] not in ("", "None", "null"):
        output += f"**Missing Info:** {result['missing_info']}\n\n"

    if result.get("sources"):
        output += "---\n**Retrieved Source Chunks:**\n"
        for i, src in enumerate(result["sources"], 1):
            preview = src[:300] + "..." if len(src) > 300 else src
            output += f"*[Chunk {i}]* {preview}\n\n"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": output})
    return history, ""


# -----------------------------------------------------------------------------
# Gradio UI — using same pattern that works in your Colab
# -----------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Wikipedia RAG Chatbot")
    gr.Markdown("Enter any Wikipedia topic below, then ask questions about it.")

    with gr.Row():
        topic_input = gr.Textbox(
            placeholder="e.g. Diabetes, Black hole, Python language",
            label="Wikipedia Topic",
            scale=4
        )
        load_btn = gr.Button("Load Topic", variant="primary", scale=1)

    status = gr.Markdown("")

    chatbot = gr.Chatbot(render_markdown=True, height=500, label="Chat")
    msg = gr.Textbox(placeholder="Ask a question about the topic...", label="Your Question")

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    gr.Examples(
        examples=[
            "What is the main definition?",
            "What are the key causes?",
            "What treatments or solutions exist?"
        ],
        inputs=msg
    )

    load_btn.click(fn=load_topic, inputs=topic_input, outputs=status)
    submit_btn.click(fn=chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(fn=chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch()
