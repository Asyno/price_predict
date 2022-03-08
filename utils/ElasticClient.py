from datetime import datetime
from typing import Dict, List

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

es = Elasticsearch("http://localhost:9200")


def index(index_name: str, doc: Dict[str, str]) -> str:
    return es.index(index=index_name, document=doc)["_id"]


def get(index_name: str, doc_id: str) -> dict:
    return es.get(index=index_name, id=doc_id)["_source"]


def generate_actions(data: pd.DataFrame):
    columns: List[str] = list(data.columns)
    for row in data.iterrows():
        doc: Dict[str] = {}
        for column in columns:
            if column == "date":
                doc["@timestamp"] = datetime.strptime(row[1][column], '%y%m%d %H:%M:%S')
            else:
                doc[column] = row[1][column]
        yield doc


def bulk_index(index_name: str, data: pd.DataFrame) -> bool:
    successes = False
    for ok, action in streaming_bulk(
            client=es, index=index_name, actions=generate_actions(data),
    ):
        successes += ok
    return successes


def create_index(index_name: str):
    if not es.indices.exists(index=index_name):
        request_body = {
            'mappings': {
                'properties': {
                    '@timestamp': {'type': 'date'}
                }}
        }
        es.indices.create(index=index_name, body=request_body)


def delete_index(index_name: str):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
