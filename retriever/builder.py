from utils.logging import logger
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from config.settings import settings

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

    def build_hybrid_retriever(self, docs: list[Document], session_id: str) -> EnsembleRetriever:
        """Build a hybrid retriever scoped to one session.

        `docs` must be that session's FULL accumulated chunk set (every upload
        so far, not just the latest one) — the caller (app.py) owns that list.
        This method always resets the session's Chroma collection and rebuilds
        it from scratch from that full set, rather than appending to whatever
        Chroma already has on disk. That's the actual fix for the Ch4-6
        accumulation bug: the old code appended to a fixed collection_name on
        every /upload with no reset, so a collection built for session A could
        never be told "start over" — and BM25 (rebuilt fresh from `docs` here
        too) only ever saw the latest file, while Chroma silently kept every
        prior one. Rebuilding both from the same authoritative list keeps them
        consistent with each other by construction.
        """
        collection_name = f"{settings.CHROMA_COLLECTION_NAME}_{session_id}"

        # Reset: drop any existing collection for this session before repopulating.
        # Safe on a session's first upload too (get_or_create under the hood).
        Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DB_DIR,
        ).delete_collection()

        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=settings.CHROMA_DB_DIR,
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = settings.VECTOR_SEARCH_K

        return EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=settings.HYBRID_RETRIEVER_WEIGHTS,
        )