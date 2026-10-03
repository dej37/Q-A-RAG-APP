# Document Q&A with RAG (LangChain + Mistral)

A retrieval-augmented generation (RAG) app that answers questions about a PDF and cites the pages it used. I built it end to end on a free stack: Mistral's free API tier, Chroma, and LangChain, with a FastAPI backend, a Docker image, and a Streamlit page to ask questions in.

I started it after finishing DataCamp's LangChain and RAG courses. I wanted to build one myself with no course guiding me, and to take it from a notebook prototype to something that runs as a service.

## What it does

You give it a PDF and ask a question. It finds the most relevant parts of the document, hands them to a language model, and returns an answer with page citations like `[Page 3]`. If the answer isn't in the document, it says it doesn't know instead of making something up.

I tested it on the arXiv paper *Recurrent Neural Networks (RNNs): A Gentle Introduction and Overview*, asking conceptual questions such as why LSTMs help with vanishing gradients. I checked the answers and page numbers by hand against the paper.

## How it works

```
Indexing (once):  PDF -> pages -> chunks -> Mistral embeddings -> Chroma database

Answering:        question -> top 4 matching chunks -> prompt with [Page N] labels
                  -> Mistral chat model -> answer with citations
```

- **Loading:** `PyPDFLoader`, one document per page, keeping the page number as metadata.
- **Chunking:** `RecursiveCharacterTextSplitter`, 1000 characters per chunk, 200 overlap.
- **Embeddings and storage:** `mistral-embed`, saved to a local Chroma database in `chroma_db/`.
- **Retrieval:** the 4 chunks closest in meaning to the question.
- **Generation:** `open-mistral-nemo` answers using only the retrieved context and cites page numbers.
- **Chain:** written with LangChain's pipe syntax: `retriever | format_docs`, `RunnablePassthrough`, `PromptTemplate`, model, `StrOutputParser`.

## What I built, in order

1. **Notebook prototype.** Load, chunk, embed, retrieve, and answer, step by step in Jupyter, checking the output of each step.
2. **Modular code.** Split the notebook into one file per step under `src/`, then wired them together in `pipeline.py`. Settings (paths, chunk size, model names) live in `src/config.py`.
3. **API.** Wrapped the chain in a FastAPI app (`api.py`) with a `/ask` endpoint.
4. **Docker.** Wrote a Dockerfile and `.dockerignore`, built the image, and ran it locally. The API key is kept out of the image and passed in at run time.
5. **Streamlit UI.** A small page (`app.py`) with a text box and an Ask button that calls the chain directly.

## Project structure

```
rag-project/
├── data/              # the PDF
├── src/
│   ├── config.py      # paths, chunk settings, model names
│   ├── loader.py      # load_pdf
│   ├── chunker.py     # chunk
│   ├── vectorstore.py # vectorize, load_vectorstore
│   ├── retriever.py   # get_retriever
│   └── generator.py   # prompt template and chain
├── pipeline.py        # wires the steps together
├── api.py             # FastAPI app with the /ask endpoint
├── app.py             # Streamlit page
├── Dockerfile
├── requirements.txt
```

## Run it locally

1. Create and activate a virtual environment (Python 3.12 recommended):

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Mac/Linux
   ```

2. Install the packages:

   ```
   pip install -r requirements.txt
   pip install streamlit requests  # only needed for the Streamlit page
   ```

3. Create a `.env` file with a free Mistral API key from console.mistral.ai. No spaces around `=`, no quotes:

   ```
   MISTRAL_API_KEY=your_key_here
   ```

4. Put a PDF in `data/` and set its filename in `src/config.py` (`PDF_PATH`).

5. Start the Streamlit page:

   ```
   streamlit run app.py
   ```

   Or run the API instead:

   ```
   uvicorn api:app --reload
   ```

   Then open http://127.0.0.1:8000/docs and try the `/ask` endpoint.

On the first run, if `chroma_db/` doesn't exist, the PDF is chunked and embedded automatically. Later runs load the saved database without calling the embedding API again. To index a different PDF, delete the `chroma_db/` folder first.

## Run the API with Docker

Run the app locally once first so `chroma_db/` exists and gets copied into the image. Then:

```
docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app
```

Open http://localhost:8000/docs. The API key is not baked into the image, because `.env` is excluded by `.dockerignore`.

## Problems I ran into

- **Rate limits.** `mistral-small-latest` kept returning 429 errors on the free tier, even for a one-word prompt. Switching to `open-mistral-nemo` fixed it.
- **Duplicate chunks.** I re-ran the cell that builds the database, and Chroma stored every chunk twice (92 instead of 46). I fixed it by rebuilding the collection once and loading the saved database from then on instead of rebuilding it.
- **RAGAS didn't install.** I wanted to evaluate with RAGAS, but the install failed on Python 3.14 because one of its dependencies has no prebuilt Windows package and needs a C++ compiler. I skipped it for now.
- **Docker and `.env`.** `python-dotenv` accepts a space before the `=`, but Docker's `--env-file` rejects it, so the container wouldn't start until I removed the space.
- **Committed junk files.** `__pycache__` files ended up in Git. I removed them from tracking and added them to `.gitignore`.

## Known limitations

- **No conversation memory.** Each question is independent, so follow-ups like "what about the second one?" don't work. It's a document Q&A tool, not yet a chatbot.
- **Equations don't extract well from PDFs.** Math comes out as garbled text, so it works better on conceptual questions than on formulas.
- **No automated evaluation yet.** I checked answers by hand against the source paper.
- **Not deployed yet.** It runs locally and in Docker.

## Next steps

- Evaluation (retrieval hit rate, answer faithfulness)
- Conversation history, so it works as a chatbot
- Deployment (Render for the API, Streamlit Community Cloud for the page)
- Support for multiple documents