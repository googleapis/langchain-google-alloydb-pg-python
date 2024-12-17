#!/usr/bin/env python

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

# [START pinecone_get_client]
from pinecone import Index, Pinecone, ServerlessSpec


def get_client(pinecone_api_key: str) -> Pinecone:
    pc = Pinecone(
        api_key=pinecone_api_key,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    print("Pinecone client initiated.")
    return pc


# [END pinecone_get_client]

# [START pinecone_get_index]
from pinecone import Pinecone, ServerlessSpec


def get_index(client: Pinecone, index_name: str = "index-name") -> Index:
    index = client.Index(index_name)

    print("Pinecone index reference initiated.")
    return index


# [END pinecone_get_index]


# [START pinecone_get_all_ids]
def get_all_ids(index: Index, namespace="") -> list[str]:
    results = index.list_paginated(prefix="", namespace=namespace)
    ids = [v.id for v in results.vectors]
    while results.pagination is not None:
        pagination_token = results.pagination.next
        results = index.list_paginated(prefix="", pagination_token=pagination_token)
        ids.extend([v.id for v in results.vectors])

    print("Pinecone client fetched all ids from index.")

    return ids


# [END pinecone_get_all_ids]


# [START pinecone_get_all_data]
def get_all_data(
    index: Index, ids: list[str]
) -> tuple[list[str], list[str], list[list[float]], list[dict]]:
    all_data = index.fetch(ids=ids)
    ids = []
    embeddings = []
    contents = []
    metadatas = []
    for doc in all_data["vectors"].values():
        ids.append(doc["id"])
        embeddings.append(doc["values"])
        contents.append(str(doc["metadata"]))
        metadata = doc["metadata"]
        metadatas.append(metadata)

    print("Pinecone client fetched all data from index.")
    return ids, contents, embeddings, metadatas


# [END pinecone_get_all_data]


if __name__ == "__main__":
    client = get_client(
        pinecone_api_key=sys.argv[1],
    )
    index = get_index(
        client=client,
        index_name=sys.argv[2],
    )
    ids = get_all_ids(
        index=index,
    )
    ids, content, embeddings, metadatas = get_all_data(index=index, ids=ids)
    print(f"Downloaded {len(ids)} values from Pinecone.")
