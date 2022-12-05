import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.array.milvus import DocumentArrayMilvus

from executor import MilvusIndexer


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.random.rand(128)),
            Document(id='doc2', embedding=np.random.rand(128)),
            Document(id='doc3', embedding=np.random.rand(128)),
            Document(id='doc4', embedding=np.random.rand(128)),
            Document(id='doc5', embedding=np.random.rand(128)),
            Document(id='doc6', embedding=np.random.rand(128)),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', text='modified', embedding=np.random.rand(128)),
        ]
    )


def test_init(docker_compose):
    indexer = MilvusIndexer(collection_name='test1')

    assert isinstance(indexer._index, DocumentArrayMilvus)
    assert indexer._index._config.collection_name == 'test1'
    assert indexer._index._config.port == 19530
    assert indexer._index._config.host == 'localhost'
    assert indexer._index._config.distance == 'IP'
    assert indexer._index._config.n_dim == 128


def test_index(docs, docker_compose):
    indexer = MilvusIndexer()
    indexer.index(docs)

    assert len(indexer._index) == len(docs)


def test_delete(docs, docker_compose):
    indexer = MilvusIndexer()
    indexer.index(docs)

    ids = ['doc1', 'doc2', 'doc3']
    indexer.delete({'ids': ids})
    assert len(indexer._index) == len(docs) - 3
    for doc_id in ids:
        assert doc_id not in indexer._index


def test_update(docs, update_docs, docker_compose):
    # index docs first
    indexer = MilvusIndexer()
    indexer.index(docs)
    assert_document_arrays_equal(indexer._index, docs)

    # update first doc
    indexer.update(update_docs)
    assert indexer._index['doc1'].text == 'modified'


def test_fill_embeddings(docker_compose):
    indexer = MilvusIndexer(distance='L2', n_dim=1)

    indexer.index(DocumentArray([Document(id='a', embedding=np.array([1]))]))
    search_docs = DocumentArray([Document(id='a')])
    indexer.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None
    assert (search_docs['a'].embedding == np.array([1])).all()

    with pytest.raises(KeyError, match='b'):
        indexer.fill_embedding(DocumentArray([Document(id='b')]))


def test_filter(docker_compose):
    n_dim = 3
    indexer = MilvusIndexer(n_dim=n_dim, columns={'price': 'float'})

    docs = DocumentArray([Document(id=f'doc{i}', tags={'price': i}) for i in range(10)])
    indexer.index(docs)

    max_price = 3
    filter_ = f'price <= {max_price}'

    result = indexer.filter(parameters={'filter': filter_})

    assert len(result) == 4
    assert result[-1].tags['price'] == max_price


def test_persistence(docs, docker_compose):
    name = 'test_persistence'
    indexer1 = MilvusIndexer(collection_name=name, distance='L2', n_dim=128)
    indexer1.index(docs)
    indexer2 = MilvusIndexer(collection_name=name, distance='L2', n_dim=128)

    assert_document_arrays_equal(indexer2._index, docs)


@pytest.mark.parametrize('metric', ['IP', 'L2'])
def test_search(metric, docs, docker_compose):
    # test general/normal case
    indexer = MilvusIndexer(distance='L2', n_dim=128)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query)

    for doc in query:
        similarities = [t['score'].value for t in doc.matches[:, 'scores']]
        assert sorted(similarities) == similarities
        assert len(similarities) == len(docs)


@pytest.mark.parametrize('limit', [1, 2, 3])
def test_search_with_match_args(docs, limit, docker_compose):
    indexer = MilvusIndexer(
        distance='L2',
        match_args={'limit': limit}
    )
    indexer.index(docs)
    assert 'limit' in indexer._match_args.keys()
    assert indexer._match_args['limit'] == limit

    query = DocumentArray([Document(embedding=np.random.rand(128))])
    indexer.search(query)

    assert len(query[0].matches) == limit

    docs[0].tags['price'] = 1.0
    docs[1].tags['price'] = 1.2
    docs[2].tags['price'] = 3.9
    docs[3].tags['price'] = 2.0
    docs[4].tags['price'] = 0.4
    docs[5].tags['price'] = 4.6

    indexer = MilvusIndexer(
        columns={'price': 'float'},
        match_args={'filter': f'price <= {2.5}', 'limit': limit},
    )
    indexer.index(docs)
    indexer.search(query)

    assert len(query[0].matches) == limit
    for match in query[0].matches:
        assert match.tags['price'] <= 2.5


def test_clear(docs, docker_compose):
    indexer = MilvusIndexer()
    indexer.index(docs)
    assert len(indexer._index) == 6
    indexer.clear()
    assert len(indexer._index) == 0


def test_columns(docker_compose):
    n_dim = 3
    indexer = MilvusIndexer(n_dim=n_dim, columns={'price': 'float'})
    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )
    indexer.index(docs)
    assert len(indexer._index) == 10
