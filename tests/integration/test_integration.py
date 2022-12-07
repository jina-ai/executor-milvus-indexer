import multiprocessing
import numpy as np

from docarray import Document, DocumentArray
from jina import Client, Flow

from executor import MilvusIndexer


def _send_request(gateway_port, protocol):
    c = Client(host='localhost', port=gateway_port, protocol=protocol)
    c.post(
        on='/index',
        inputs=[
            Document(id='a', embedding=np.array([1, 3])),
            Document(id='b', embedding=np.array([1, 1])),
            Document(id='c', embedding=np.array([3, 1])),
            Document(id='d', embedding=np.array([2, 3])),
        ],
    )

    res_search = c.post(
        on='/search',
        inputs=[Document(embedding=np.array([1, 1]))],
    )
    assert res_search[0].matches[0].id == 'b'


def _create_flow():
    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={'n_dim': 2},
    )
    
    with f:
        q = multiprocessing.Process(
            target=_send_request, args=(f.port, 'grpc')
        )
        q.start()
        q.join()

        assert q.exitcode == 0

#
# def test_flow(docker_compose):
#     p = multiprocessing.Process(target=_create_flow)
#     p.start()
#     p.join()
#
#     assert p.exitcode == 0


# def test_flow_2(docker_compose):
#     f = Flow().add(
#         uses=MilvusIndexer,
#         uses_with={'n_dim': 2},
#     )
#
#     with f:
#         f.post(
#             on='/index',
#             inputs=[
#                 Document(id='a', embedding=np.array([1, 3])),
#                 Document(id='b', embedding=np.array([1, 1])),
#                 Document(id='c', embedding=np.array([3, 1])),
#                 Document(id='d', embedding=np.array([2, 3])),
#             ],
#         )
#
#         res_search = f.post(
#             on='/search',
#             inputs=[Document(embedding=np.array([1, 1]))],
#         )
#         assert res_search[0].matches[0].id == 'b'


def test_reload_keep_state(docker_compose):
    docs = DocumentArray([Document(embedding=np.random.rand(3)) for _ in range(2)])
    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={'collection_name': 'test_keep_state3', 'n_dim': 3},
    )

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)
        assert len(first_search[0].matches) == 2

    with f:
        second_search = f.search(inputs=docs)
        assert len(second_search[0].matches) == 2


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
            indexed_docs = f.search(doc_query, parameters={'filter': f'price <= {threshold}'})

            assert len(indexed_docs[0].matches) > 0
