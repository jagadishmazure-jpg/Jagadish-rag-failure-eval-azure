"""Create the index or table and load the w80 corpus into whichever stores are configured.

    AZURE_SEARCH_ENDPOINT   Azure AI Search (Entra ID, index tessmoor-policies)
    COSMOS_ENDPOINT         plus COSMOS_DATABASE and COSMOS_CONTAINER
    PGVECTOR_DSN            local compose, or Azure with PGVECTOR_ENTRA=1

Unset stores are skipped. Nothing here runs unless you execute the script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail import data  # noqa: E402
from ragfail.vectorstores import cosmos_items, pg_connect  # noqa: E402


def load() -> list[str]:
    retriever = data.retriever("w80")
    done = []
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    if endpoint:
        from ragfail.azure_search import upload

        upload(endpoint, retriever.chunks, retriever.emb)
        done.append("azure_ai_search")
    cosmos = os.environ.get("COSMOS_ENDPOINT")
    if cosmos:
        from azure.cosmos import CosmosClient
        from azure.identity import DefaultAzureCredential

        client = CosmosClient(cosmos, credential=DefaultAzureCredential())
        container = client.get_database_client(os.environ["COSMOS_DATABASE"]).get_container_client(os.environ["COSMOS_CONTAINER"])
        for item in cosmos_items(retriever.chunks, retriever.emb):
            container.upsert_item(item)
        done.append("cosmos_nosql")
    dsn = os.environ.get("PGVECTOR_DSN")
    if dsn:
        sql = (ROOT / "infra" / "pgvector" / "init.sql").read_text()
        with pg_connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql)
            for item in cosmos_items(retriever.chunks, retriever.emb):
                cur.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, title, text, status, doc_type, district, topic, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding, "
                    "status = EXCLUDED.status, doc_type = EXCLUDED.doc_type",
                    (item["chunk_id"], item["doc_id"], item["title"], item["text"], item["status"], item["doc_type"],
                     item["district"], item["topic"], item["vector"]),
                )
            conn.commit()
        done.append("pgvector")
    return done


def main() -> int:
    done = load()
    print("loaded: " + (", ".join(done) if done else "nothing (set AZURE_SEARCH_ENDPOINT, COSMOS_ENDPOINT, or PGVECTOR_DSN)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
