import numpy as np
from docarray import Document, DocumentArray
from jina import Flow

from executor import MilvusIndexer


def test_filtering(docker_compose):
    n_dim = 3

    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={
            'n_dim': n_dim,
            'columns': {'price': 'float'},
        },
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    with f:
        f.index(docs)

        for threshold in [10, 20, 30]:

            doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
            indexed_docs = f.search(
                doc_query, parameters={'filter': f'price <= {threshold}'}
            )

            assert len(indexed_docs[0].matches) > 0
