import numpy as np
from docarray import Document
from executor import MilvusIndexer
from jina import Flow


def test_replicas(docker_compose):
    n_dim = 10
    name = 'test_replicas1dsdg'

    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={
            'collection_name': name,
            'n_dim': n_dim,
        },
    )

    docs_index = [
        Document(id=str(i), embedding=np.random.random(n_dim)) for i in range(10)
    ]

    docs_query = docs_index[:4]

    with f:
        print('in with f')
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=MilvusIndexer,
        uses_with={'collection_name': name, 'n_dim': n_dim},
        replicas=4,
    )

    with f_with_replicas:
        print('in with f replicas')
        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas


def test_replicas_reindex(docker_compose):
    n_dim = 10
    name = 'test_reindex'

    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={
            'collection_name': name,
            'n_dim': n_dim,
        },
    )

    docs_index = [
        Document(id=f'd{i}', embedding=np.random.random(n_dim)) for i in range(10)
    ]

    docs_query = docs_index[:4]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=MilvusIndexer,
        uses_with={'collection_name': name, 'n_dim': n_dim},
        replicas=4,
    )

    with f_with_replicas:
        f_with_replicas.post(on='/index', inputs=docs_index, request_size=1)

        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas
