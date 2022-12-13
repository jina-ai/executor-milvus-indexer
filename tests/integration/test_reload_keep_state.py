import numpy as np
from docarray import Document, DocumentArray
from jina import Flow

from executor import MilvusIndexer


def test_reload_keep_state(docker_compose):
    docs = DocumentArray([Document(embedding=np.random.rand(3)) for _ in range(2)])
    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={'collection_name': 'test_keep_state', 'n_dim': 3},
    )

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)
        assert len(first_search[0].matches) == 2

    with f:
        second_search = f.search(inputs=docs)
        assert len(second_search[0].matches) == 2
