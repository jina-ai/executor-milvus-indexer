from typing import Optional, Union, Dict, List, Tuple

from docarray import DocumentArray
from jina import Executor, requests

from jina.logging.logger import JinaLogger


class MilvusIndexer(Executor):
    """MilvusIndexer indexes Documents into a Milvus server using DocumentArray with `storage='milvus'`"""

    def __init__(
        self,
        collection_name: str = None,
        n_dim: int = 128,
        host: str = 'localhost',
        port: Optional[Union[str, int]] = 19530,  # 19530 for gRPC, 9091 for HTTP
        distance: str = 'IP',  # metric_type in milvus
        index_type: str = 'HNSW',
        index_params: Dict = None,
        collection_config: Dict = None,
        serialize_config: Dict = None,
        consistency_level: str = 'Session',
        batch_size: int = -1,
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
        match_args: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._match_args = match_args or {}

        self._index = DocumentArray(
            storage='milvus',
            config={
                'collection_name': collection_name,
                'n_dim': n_dim,
                'host': host,
                'port': port,
                'distance': distance,
                'index_type': index_type,
                'index_params': index_params or {'M': 4, 'efConstruction': 200},
                'collection_config': collection_config or {},
                'serialize_config': serialize_config or {},
                'consistency_level': consistency_level,
                'batch_size': batch_size,
                'columns': columns,
                'list_like': False,
            },
        )

        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        """Index new documents
        :param docs: the Documents to index
        """
        with self._index:
            self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: Dictionary to define the `filter` that you want to use.
        :param kwargs: additional kwargs for the endpoint

        """

        if parameters is None:
            parameters = {}

        match_args = (
            {**self._match_args, **parameters}
            if parameters is not None
            else self._match_args
        )
        docs.match(self._index, **match_args)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters of the request

        Keys accepted:
            - 'ids': List of Document IDs to be deleted
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update existing documents
        :param docs: the Documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """Fill embedding of Documents by id

        :param docs: DocumentArray to be filled with Embeddings from the index
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/filter')
    def filter(self, parameters: Dict, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray` using Redis: https://docarray.jina.ai/advanced/document-store/redis/#search-by-filter-query
        :param parameters: parameters of the request, containing the `filter` query
        """
        return self._index.find(filter=parameters.get('filter', None))

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index"""
        self._index.clear()

    def close(self) -> None:
        super().close()
        del self._index
