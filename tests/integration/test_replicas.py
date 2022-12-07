import numpy as np
from docarray import Document
from executor import MilvusIndexer
from jina import Flow


def test_replicas(docker_compose):
    n_dim = 1
    name = 'test_replicas'

    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={
            'collection_name': name,
            'n_dim': n_dim,
        },
    )

    docs_index = [
        Document(id=str(i), embedding=np.random.random(n_dim)) for i in range(4)
    ]

    docs_query = docs_index[:2]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=MilvusIndexer,
        uses_with={'collection_name': name, 'n_dim': n_dim},
        replicas=2,
    )

    with f_with_replicas:
        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas


def test_replicas_reindex(docker_compose):
    n_dim = 1
    name = 'test_reindex22d1dshhbjklhjsksqa'

    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={
            'collection_name': name,
            'n_dim': n_dim,
        },
    )

    docs_index = [
        Document(id=f'd{i}', embedding=np.random.random(n_dim)) for i in range(4)
    ]

    docs_query = docs_index[:2]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=MilvusIndexer,
        uses_with={'collection_name': name, 'n_dim': n_dim},
        replicas=2,
    )

    with f_with_replicas:
        f_with_replicas.post(on='/index', inputs=docs_index, request_size=1)

        docs_with_replicas = f_with_replicas.post(on='/search', inputs=docs_query, request_size=1)
        docs_with_replicas = sorted(
            docs_with_replicas,
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas
