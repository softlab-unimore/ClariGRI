import torch
import os
import numpy as np
import json
import functools
import logging
import time
from datetime import datetime
from typing import List, Union, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from connectors import PgVectorConnector
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.vector_store")


class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str, max_seq_length: int = 8192):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length

    def embed_documents(self, texts: list) -> list:
        return self.model.encode(texts)

    def embed_query(self, query: str) -> list:
        return self.model.encode([query])[0]  # Returns the embedding for a single query


class Handler(object):
    def __init__(self, args):
        super(object, self).__init__()

    @staticmethod
    def hash_doc(doc):
        content = doc.page_content
        metadata = doc.metadata

        input_hash = content + ''.join(list(map(lambda x: str(x), list(metadata.values()))))
        return str(hash(input_hash))


class VectorStoreHandler(Handler):
    def __init__(self, args):
        super(VectorStoreHandler, self).__init__(args)
        # self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_name = args["model_name"]

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.embeddings = self.get_embeddings(self.model_name, device)
        # self.embeddings = CustomHuggingFaceEmbeddings(model_name=args["model_name"])
        self.pgconnector = PgVectorConnector()

    @functools.cache
    def get_embeddings(self, model_name, device="cpu"):
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

    @functools.cache
    def get_vector_store(self, collection_name="coll"):

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=self.pgconnector.get_connection(),
            use_jsonb=True,
        )

        return self.vector_store

    def load_docs_in_vector_store(self, docs, company_name=None, company_sectors=None):
        """
        docs: list[str] -> [
            {
                "page_content": "...",
                "metadata": {...}
            }
        ]
        """
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Adding {len(docs)} documents into the vector store...")
        hashes = []

        # add model_name to docs metadata
        for i in range(len(docs)):
            docs[i].metadata["model_name"] = self.model_name
            if company_name:
                docs[i].metadata["company_name"] = company_name
            if company_sectors:
                docs[i].metadata["company_sectors"] = company_sectors

        for doc in docs:
            hashed_input = self.hash_doc(doc)
            hashes.append(hashed_input)

        conn = self.pgconnector.start_db_connection()
        existing_ids = self.pgconnector.get_existing_ids(conn, hashes, table_name=os.environ['POSTGRES_EMB_TABLE_NAME'])
        self.pgconnector.close_db_connection(conn)

        allowed_docs = []
        allowed_hashes = []
        unallowed_docs = []  # for debugging

        for hash, doc in zip(hashes, docs):
            if hash in existing_ids:
                unallowed_docs.append((doc.metadata["source"], doc.metadata["page"]))
            else:
                allowed_docs.append(doc)
                allowed_hashes.append(hash)

        if len(unallowed_docs) > 0:
            logger.info(
                f"[{datetime.now()}] The following documents {unallowed_docs} are already in the db. Skipping...")

        for hash, doc in tqdm(zip(allowed_hashes,
                                  allowed_docs)):  # processing docs and hashes one-by-one to prevent db connection drops (https://docs.sqlalchemy.org/en/20/errors.html#error-e3q8)
            self.vector_store.add_documents([doc], ids=[hash])
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Added {len(allowed_docs)} documents in {end_time - start_time} seconds")

    def delete_from_vector_store(self, ids: Union[list[str], str], collection_name="coll"):
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Removing {ids} documents from the vector store...")
        if isinstance(ids, str):
            if ids != "all":
                raise ValueError(f"{ids} is not a valid index. Please provide a list of indices or the \"all\" string")
            self.vector_store.delete_collection(collection_name)
        else:
            self.vector_store.delete(ids=ids)
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Removed {ids} documents in {end_time - start_time} seconds")

    @functools.cache
    def query_by_similarity(self, query, k=5, filters=(), with_scores=False):
        d_filter = {}
        for i in range(len(filters)):
            d_filter[filters[i][0]] = filters[i][1]

        if with_scores:
            result = self.vector_store.similarity_search_with_score(query, k=k, filter=d_filter)
        else:
            result = self.vector_store.similarity_search(query, k=k, filter=d_filter)

        return result

    @functools.cache
    def query_by_similarity_with_sectors(self, query: str, filters: tuple, k: int = 5, with_scores=False):
        """
        Performs a similarity search only on chunks belonging to the specified sectors.
        Also filters by model_name and source if present in the filters.
        """
        sectors = None

        # Filter extraction
        for key, value in filters:
            if key == "sectors":
                sectors = value

        # Recover chunks only for specified sectors
        conn = self.pgconnector.start_db_connection()
        docs, docs_lowered = self.pgconnector.get_all_chunks_vectors(conn, sectors)
        self.pgconnector.close_db_connection(conn)
        doc_embeddings = []
        doc_metadata = []
        doc_contents = []

        for d in docs_lowered:
            emb_str = d.metadata.get("embedding")
            if emb_str is None:
                continue
            # Convert JSON string to list of floats
            emb = np.array(json.loads(emb_str), dtype=float)
            doc_embeddings.append(emb)
            doc_metadata.append(d.metadata)
            doc_contents.append(d.page_content)

        from langchain_huggingface import HuggingFaceEmbeddings

        # Calculate the query embedding
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        query_emb = np.array(embedder.embed_query(query))

        # Calculate cosine similarity
        doc_embeddings = np.array(doc_embeddings)
        norm_docs = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norm_query = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(norm_docs, norm_query)
        
        # Sort by decreasing similarity
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:k]

        # Builds the final results
        results = []
        for idx in top_indices:
            doc = docs_lowered[idx]
            if with_scores:
                results.append((doc, float(similarities[idx])))
            else:
                results.append(doc)
        return results


class CustomTFIDFRetriever(TFIDFRetriever):
    vectorizer: Any
    """TF-IDF vectorizer."""
    docs: List[Document]
    """Documents."""
    tfidf_array: Any
    """TF-IDF array."""
    k: int = 4
    """Number of documents to return."""

    class Config:
        arbitrary_types_allowed = True


class SparseStoreHandler(Handler):
    def __init__(self, args):
        super(SparseStoreHandler, self).__init__(args)
        self.switch_model = {
            "tf_idf": CustomTFIDFRetriever
        }
        self.model_name = args["syn_model_name"]
        self.pgconnector = PgVectorConnector()

    def load_docs_in_sparse_store(self, docs, company_name=None, company_sectors=None):
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Adding {len(docs)} documents into the sparse store...")
        hashes = []

        # add model_name to docs metadata
        for i in range(len(docs)):
            docs[i].metadata["model_name"] = self.model_name
            if company_name:
                docs[i].metadata["company_name"] = company_name
            if company_sectors:
                docs[i].metadata["company_sectors"] = company_sectors

        for doc in docs:
            hashed_input = self.hash_doc(doc)
            hashes.append(hashed_input)

        conn = self.pgconnector.start_db_connection()
        existing_ids = self.pgconnector.get_existing_ids(conn, hashes,
                                                         table_name=os.environ['POSTGRES_SPARSE_TABLE_NAME'])

        allowed_docs = []
        allowed_hashes = []
        unallowed_docs = []  # for debugging

        for hash, doc in zip(hashes, docs):
            if hash in existing_ids:
                unallowed_docs.append((doc.metadata["source"], doc.metadata["page"]))
            else:
                allowed_docs.append(doc)
                allowed_hashes.append(hash)

        if len(unallowed_docs) > 0:
            logger.info(
                f"[{datetime.now()}] The following documents {unallowed_docs} are already in the db. Skipping...")

        for hash, doc in tqdm(zip(allowed_hashes,
                                  allowed_docs)):  # processing docs and hashes one-by-one to prevent db connection drops (https://docs.sqlalchemy.org/en/20/errors.html#error-e3q8)
            elements_to_add = (
                hash,
                '.'.join(doc.metadata["source"].split("/")[-1].split(".")[:-1]),
                doc.metadata["source"],
                doc.page_content,
                doc.metadata["page"],
                self.model_name,
                doc.metadata.get("company_name"),
                doc.metadata.get("company_sectors")
            )

            self.pgconnector.add_page(conn, elements_to_add)

        self.pgconnector.close_db_connection(conn)
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Added {len(allowed_docs)} documents in {end_time - start_time} seconds")

    @functools.cache
    def query_by_similarity(self, query, source, k=5, with_scores=False):
        conn = self.pgconnector.start_db_connection()
        docs, docs_lowered = self.pgconnector.get_pages(conn, source)

        self.pgconnector.close_db_connection(conn)

        retriever = self.switch_model[self.model_name].from_documents(docs_lowered)

        retriever.k = k
        results = retriever.invoke(query)

        if with_scores:
            query_vec = retriever.vectorizer.transform([query])
            results_array = cosine_similarity(retriever.tfidf_array, query_vec).reshape((-1,))
            scores = sorted(results_array)[-retriever.k:][::-1]

            result_with_scores = []
            for result, score in zip(results, scores):
                result_with_scores.append((result, score))
            results = result_with_scores

        return results

    @functools.cache
    def query_by_similarity_with_sectors(self, query: str, filters: tuple, k: int = 5, with_scores=False):
        """
        Search all chunks by filtering for specified sectors.
        """
        sectors = None

        for key, value in filters:
            if key == "sectors":
                sectors = value

        conn = self.pgconnector.start_db_connection()
        docs, docs_lowered = self.pgconnector.get_all_chunks(conn, sectors)
        self.pgconnector.close_db_connection(conn)

        retriever = self.switch_model[self.model_name].from_documents(docs_lowered)
        retriever.k = k

        query_vec = retriever.vectorizer.transform([query])
        docs_vec = retriever.tfidf_array
        similarities = cosine_similarity(docs_vec, query_vec).reshape(-1)

        top_indices = np.argsort(similarities)[::-1][:k]

        if with_scores:
            results = [(docs_lowered[i], float(similarities[i])) for i in top_indices]
        else:
            results = [docs_lowered[i] for i in top_indices]

        return results


class EnsembleRetrieverHandler(SparseStoreHandler, VectorStoreHandler):
    def __init__(self, args):
        super(EnsembleRetrieverHandler, self).__init__(args)
        self.lmbd = args["lambda"]

    def combine_results(self, semantic_results, syntactic_results, k=5, lmbd=.3):
        results = {}
        res_debug = {}
        for i, r in enumerate([semantic_results, syntactic_results]):
            for j, el in enumerate(r):

                # semantic and syntactic Documents have different model_names
                # we change them to an empty string to allow Document matching

                el[0].metadata["model_name"] = ""
                if i == 0:
                    el[0].page_content = el[0].page_content.lower()

                hashed_key = self.hash_doc(el[0])
                score = -el[1] if i == 0 else el[1] * lmbd
                if hashed_key not in results.keys():
                    results[hashed_key] = [el[0], score]
                    res_debug[hashed_key] = [score, 0]
                else:
                    results[hashed_key][1] += score
                    res_debug[hashed_key][1] = score

        values = results.values()
        values = sorted(values, key=lambda x: x[1], reverse=True)[:k]

        return [el[0] for el in values]

    @functools.cache
    def query_by_similarity(self, query, filters, k=5):
        if len(filters) == 0:
            raise ValueError(f"\"filters\" param cannot be empty")
        if filters[0][0] != "source":
            raise ValueError(f"filter \"source\" is expected in first position of filters param")
        if not filters[0][1] or not isinstance(filters[0][0], str):
            raise ValueError(f"the value of the \"source\" param must be a string")

        self.get_vector_store()
        semantic_results = VectorStoreHandler.query_by_similarity(self, query, k=5, filters=filters,
                                                                  with_scores=True)
        syntactic_results = SparseStoreHandler.query_by_similarity(self, query, source=filters[0][1], k=5,
                                                                   with_scores=True)

        results = self.combine_results(semantic_results, syntactic_results, k=k, lmbd=self.lmbd)

        return results

    @functools.cache
    def query_by_similarity_with_sectors(self, query: str, filters: tuple, k: int = 5):
        """
        Ensemble retrieval function with filtering by sectors.
        filters: tuples of pairs (key, value), e.g. ((‘sectors’, (‘sector1’,‘sector2’)), (“model_name”,‘model’))
        """

        sectors = []
        model_name = None
        for key, value in filters:
            if key == "sectors":
                sectors.extend(value)
            elif key == "model_name":
                model_name = value

        vector_filters = []
        if sectors:
            vector_filters.append(("company_sectors", sectors))
        if model_name:
            vector_filters.append(("model_name", model_name))

        semantic_results = VectorStoreHandler.query_by_similarity_with_sectors(
            self,
            query,
            filters=filters,
            k=k,
            with_scores=True
        )
       
        syntactic_results = SparseStoreHandler.query_by_similarity_with_sectors(
            self,
            query,
            filters=filters,
            k=k,
            with_scores=True
        )

        results = self.combine_results(semantic_results, syntactic_results, k=k, lmbd=self.lmbd)

        return results
