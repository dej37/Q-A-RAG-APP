from dotenv import load_dotenv

load_dotenv()

PDF_PATH = 'C:\\Users\\Top Prix\\rag-project\\data\\RNN.pdf'

CHROMA_DIR = "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "open-mistral-nemo"