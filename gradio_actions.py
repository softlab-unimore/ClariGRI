import os
import gradio as gr
import pandas as pd
from connectors import PgVectorConnector


def get_docs_from_db():
    connector = PgVectorConnector()
    conn = connector.start_db_connection()

    query = f"SELECT DISTINCT title FROM {os.environ['POSTGRES_SPARSE_TABLE_NAME']} ORDER BY title;"
    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchall()

    connector.close_db_connection(conn)
    return [r[0] for r in result]


def get_sectors_from_db():
    connector = PgVectorConnector()
    conn = connector.start_db_connection()

    query = f"""
            SELECT DISTINCT unnest(company_sectors) 
            FROM {os.environ['POSTGRES_SPARSE_TABLE_NAME']}
            WHERE company_sectors IS NOT NULL;
        """

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchall()

    sectors = [row[0] for row in result if row[0]]
    connector.close_db_connection(conn)
    return sectors


def update_docs_list():
    docs = get_docs_from_db()
    return gr.update(choices=docs, value=[])


def update_sectors_list():
    sectors = get_sectors_from_db()
    return gr.update(choices=sectors, value=[])


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def refresh_pdf_folders():
    folders = list_pdf_folders()
    return gr.update(choices=folders, value=folders[0] if folders else None)


def refresh_docs_list():
    choices = get_docs_from_db()
    return gr.update(choices=choices, value=[])


def refresh_sectors_list():
    choices = get_sectors_from_db()
    return gr.update(choices=choices, value=[])


def list_pdf_folders():

    base = os.path.join(".", "table_dataset")
    if not os.path.exists(base):
        return []
    return [
        os.path.relpath(root, base)
        for root, dirs, files in os.walk(base)
        if root != base
    ]


def list_csv_files(pdf_basename):
    refresh_pdf_folders()
    if not pdf_basename:
        return gr.update(choices=[], value=None)
    folder = os.path.join(".", "table_dataset", pdf_basename)
    if not os.path.exists(folder):
        return gr.update(choices=[], value=None)
    csvs = [f for f in os.listdir(folder) if f.endswith(".csv")]
    return gr.update(choices=csvs, value=csvs[0] if csvs else None)


def load_csv(pdf_basename, csv_filename):
    if not pdf_basename or not csv_filename:
        return gr.update(value=pd.DataFrame(), visible=False)

    path = os.path.join(".", "table_dataset", pdf_basename, csv_filename)

    if not os.path.exists(path):
        return gr.update(value=pd.DataFrame(), visible=False)

    return gr.update(value=pd.read_csv(path, sep=';', on_bad_lines='warn', engine='python', dtype=str), visible=True)


def save_csv(pdf_basename, csv_filename, df):
    if not pdf_basename or not csv_filename:
        return "⚠️ First, select a folder and a CSV file."
    path = os.path.join(".", "table_dataset", pdf_basename, csv_filename)
    df.to_csv(path, index=False, sep=';')
    return f"✅ File {csv_filename} saved in the folder {pdf_basename}."
