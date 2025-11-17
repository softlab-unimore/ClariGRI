import os
import logging
import psycopg2
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.connector")


class PgVectorConnector:
    def __init__(self):
        username = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]
        port = os.environ["POSTGRES_PORT"]
        self.connection = f"postgresql+psycopg://{username}:{password}@localhost:{port}/{db}"

    def get_connection(self):
        return self.connection

    def start_db_connection(self):
        connection = self.connection.replace("+psycopg", "")
        return psycopg2.connect(connection)

    @staticmethod
    def close_db_connection(conn):
        conn.close()

    @staticmethod
    def get_existing_ids(conn, ids, table_name):
        formatted_ids = ', '.join(f"'{id}'" for id in ids)
        query = f"SELECT id FROM {table_name} WHERE id IN ({formatted_ids});"

        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
        return [row[0] for row in result]

    @staticmethod
    def add_page(conn, elements_to_add: tuple):
        if len(elements_to_add) != 8:
            raise ValueError(
                f"the new row to add in the \"{os.environ['POSTGRES_SPARSE_TABLE_NAME']}\" table must have 8 elements")
        # if not all([isinstance(el, str) for el in elements_to_add]):
        # raise ValueError(f"the new row to add in the \"{os.environ['POSTGRES_SPARSE_TABLE_NAME']}\" table can only have str elements")

        query = f"INSERT INTO {os.environ['POSTGRES_SPARSE_TABLE_NAME']} (id, title, source, page_content, page_nbr, model_name, company_name, company_sectors) VALUES(%s, %s, %s, %s, %s, %s, %s, %s);"

        with conn.cursor() as cur:
            cur.execute(query, elements_to_add)
            conn.commit()

    @staticmethod
    def get_pages(conn, source):
        query = f"SELECT source, page_nbr, model_name, page_content FROM {os.environ['POSTGRES_SPARSE_TABLE_NAME']} WHERE source=%s;"

        with conn.cursor() as cur:
            cur.execute(query, (source,))
            result = cur.fetchall()

        docs = []
        docs_lowered = []
        for res in result:
            doc = Document(page_content=res[-1], metadata={"page": res[1], "source": res[0], "model_name": res[2]})
            doc_lowered = Document(page_content=res[-1].lower(),
                                   metadata={"page": res[1], "source": res[0], "model_name": res[2]})
            docs.append(doc)
            docs_lowered.append(doc_lowered)

        return docs, docs_lowered

    @staticmethod
    def get_all_chunks(conn, sectors):

        sectors = list(sectors)

        query = f""" SELECT source, page_nbr, model_name, company_sectors, page_content FROM {os.environ['POSTGRES_SPARSE_TABLE_NAME']} WHERE company_sectors &&  %s"""

        with conn.cursor() as cur:
            sql = cur.mogrify(query, (sectors,))
            print("Query finale:", sql.decode())
            cur.execute(query, (sectors,))
            result = cur.fetchall()

        docs = []
        docs_lowered = []
        for res in result:
            doc = Document(page_content=res[-1], metadata={"page": res[1], "source": res[0], "model_name": res[2], "company_sectors": res[3]})
            doc_lowered = Document(page_content=res[-1].lower(), metadata={"page": res[1], "source": res[0], "model_name": res[2], "company_sectors": res[3]})
            docs.append(doc)
            docs_lowered.append(doc_lowered)

        return docs, docs_lowered

    @staticmethod
    def get_all_chunks_vectors(conn, sectors):

        """
        Returns all chunks (document + metadata + embedding)belonging to the specified sectors.
        """
        sectors = list(sectors)

        query = f"""
                   SELECT document,cmetadata, embedding
                   FROM {os.environ['POSTGRES_EMB_TABLE_NAME']}
                   WHERE  cmetadata->'company_sectors' ?| %s;
               """

        with conn.cursor() as cur:
            sql = cur.mogrify(query, (sectors,))
            print("Query finale:", sql.decode())
            cur.execute(query, (sectors,))
            results = cur.fetchall()

        docs = []
        docs_lowered = []

        for document_text, metadata, embedding in results:
            doc = Document(
                page_content=document_text,
                metadata=metadata
            )
            doc_lowered = Document(
                page_content=document_text.lower(),
                metadata=metadata.copy()
            )

            doc.metadata["embedding"] = embedding
            doc_lowered.metadata["embedding"] = embedding

            docs.append(doc)
            docs_lowered.append(doc_lowered)
        print("Totale docs_lowered:", len(docs_lowered))
        print("Totale docs:", len(docs))
        return docs, docs_lowered
