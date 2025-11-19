#!/bin/sh
set -e

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<EOSQL

CREATE TABLE IF NOT EXISTS "${POSTGRES_SPARSE_TABLE_NAME}" (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    source VARCHAR(255) NOT NULL,
    page_content TEXT NOT NULL,
    page_nbr INTEGER NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    company_name TEXT,
    sectors TEXT[]
);
EOSQL

