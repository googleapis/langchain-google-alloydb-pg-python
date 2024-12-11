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


from math import sqrt

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class FakeEmbedding(Embeddings, BaseModel):
    """Generates a test embedding vector for a given string.
    Strings with common characters are similar.
    """

    size: int

    def _get_embedding(self, txt: str) -> list[float]:

        embedding = np.zeros(self.size)
        for c in list(txt):
            embedding[ord(c) % self.size] += 1.0

        # normalize the vector so that cosine distance is always [0,1]
        normalize_factor = sqrt(2.0) * np.linalg.norm(embedding)
        normalized_embedding = embedding / normalize_factor
        return list(normalized_embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)
