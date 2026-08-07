import logging
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class HFEmbeddings(Embeddings):
    """Thin wrapper around HF's current InferenceClient, since langchain_huggingface's
    embeddings class targets the old (now-removed) api-inference.huggingface.co host."""

    def __init__(self):
        self.client = InferenceClient(provider="hf-inference", api_key=settings.HF_API_KEY)


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            vec = self.client.feature_extraction(text, model=settings.EMBEDDING_MODEL)
            if vec.ndim > 1:  # some models return per-token vectors; mean-pool to one vector per text
                vec = vec.mean(axis=0)
            embeddings.append(vec.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class RetrieverBuilder:
    def __init__(self):
        self.embeddings = HFEmbeddings()

    def build_hybrid_retriever(self, docs: list[Document]) -> EnsembleRetriever:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_DB_DIR,
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = settings.VECTOR_SEARCH_K

        return EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=settings.HYBRID_RETRIEVER_WEIGHTS,
        )