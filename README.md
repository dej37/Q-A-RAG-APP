Document Q&A with RAG (LangChain + Mistral)

A retrieval-augmented generation (RAG) app that answers questions about a PDF and cites the pages it used. It runs on a free stack (Mistral free API tier, Chroma, LangChain), is served through a FastAPI endpoint, and ships as a Docker container.

I built it to practice the full path from notebook prototype to deployable service: loading, chunking, embedding, retrieval, generation, API, and containerization.

How it works
PDF -> load pages -> split into chunks -> embed (Mistral) -> store (Chroma)

Question -> retrieve top 4 chunks -> prompt with page labels -> Mistral chat model -> answer with [Page N] citations
Loading: PyPDFLoader reads the PDF, one document per page, keeping the page number as metadata.
Chunking: RecursiveCharacterTextSplitter, 1000 characters per chunk with 200 overlap.
Embeddings and storage: mistral-embed vectors saved in a local Chroma database (chroma_db/).
Retrieval: the 4 chunks closest in meaning to the question.
Generation: open-mistral-nemo answers using only the retrieved context, cites page numbers like [Page 3], and says it doesn't know when the answer isn't in the document.
Chain: built with LangChain's pipe syntax (retriever | format_docs, RunnablePassthrough, PromptTemplate, model, StrOutputParser).
Project structure
rag-project/
├── data/              # your PDF(s)
├── src/
│   ├── config.py      # paths, chunk settings, model names
│   ├── loader.py      # load_pdf
│   ├── chunker.py     # chunk
│   ├── vectorstore.py # vectorize, load_vectorstore
│   ├── retriever.py   # get_retriever
│   └── generator.py   # prompt template and chain
├── pipeline.py        # wires the steps together
├── api.py             # FastAPI app with the /ask endpoint
├── Dockerfile
├── requirements.txt
└── rag.ipynb          # original notebook prototype
Run it locally
Create and activate a virtual environment (Python 3.12 recommended):
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Mac/Linux
Install the packages:
   pip install -r requirements.txt
Create a .env file with your Mistral API key (free key from console.mistral.ai). No spaces around =, no quotes:
   MISTRAL_API_KEY=your_key_here
Put your PDF in data/ and set its filename in src/config.py (PDF_PATH).
Start the API:
   uvicorn api:app --reload

On the first run, if chroma_db/ doesn't exist, the PDF is chunked and embedded automatically. Later runs load the saved database without calling the embedding API again. To index a different PDF, delete the chroma_db/ folder first.

Open http://127.0.0.1:8000/docs, choose /ask, click Try it out, and enter a question. Or from the terminal:
   curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is self-attention?\"}"

Response:

json
   {"answer": "... [Page 7]"}
Run with Docker

Run the app locally once first so chroma_db/ exists and gets copied into the image. Then:

docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app

The API key is not baked into the image (.env is excluded by .dockerignore). It is passed in at run time with --env-file. Then open http://localhost:8000/docs.

Known limitations
Single question in, single answer out. There is no conversation memory, so follow-ups like "what about the second one?" don't work. Chat history is a planned addition.
Math doesn't extract well from PDFs. Equations come out as garbled text, so the app works better on conceptual questions than on formulas.
Free-tier rate limits. mistral-small-latest returned 429 rate-limit errors on the free tier, so the app uses open-mistral-nemo. Change the model in src/config.py.
No automated evaluation yet. Answers were checked by hand against the source paper. Adding retrieval and answer-quality evaluation is the next step.
Roadmap
Evaluation (retrieval hit rate, answer faithfulness)
Conversation history and a chat UI (Streamlit)
Cloud deployment (Google Cloud Run)
Multiple documents