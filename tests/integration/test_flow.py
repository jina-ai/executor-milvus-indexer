import multiprocessing
import numpy as np

from docarray import Document, DocumentArray
from jina import Client, Flow

from executor import MilvusIndexer



def test_flow_2(docker_compose):
    f = Flow().add(
        uses=MilvusIndexer,
        uses_with={'n_dim': 2},
    )

    with f:
        f.post(
            on='/index',
            inputs=[
                Document(id='a', embedding=np.array([1, 3])),
                Document(id='b', embedding=np.array([1, 1])),
                Document(id='c', embedding=np.array([3, 1])),
                Document(id='d', embedding=np.array([2, 3])),
            ],
        )

        res_search = f.post(
            on='/search',
            inputs=[Document(embedding=np.array([1, 1]))],
        )
        assert res_search[0].matches[0].id == 'b'
