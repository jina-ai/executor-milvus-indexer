# MilvusIndexer








`MilvusIndexer` indexes Documents into a `DocumentArray`  using `storage='milvus'`. Underneath, the `DocumentArray`  uses 
 [Milvus](https://milvus.io/) to store and search Documents efficiently. 
The indexer relies on `DocumentArray` as a client for Milvus, you can read more about the integration here: 
https://docarray.jina.ai/advanced/document-store/milvus/


## Setup
`MilvusIndexer` requires a running Milvus service. Make sure a service is up and running and your indexer 
is configured to use it before starting to index Documents. For quick testing, you can run a containerized version 
locally using docker-compose :

```shell
docker-compose -f tests/docker-compose.yml up -d
```

Note that if you run a `Milvus` service locally and try to run the `MilvusIndexer` via `docker`, you 
have to specify `'host': 'host.docker.internal'` instead of `localhost`, otherwise the client will not be 
able to reach the service from within the container.


## Usage

### via Docker image (recommended)

```python
from jina import Flow
from docarray import Document
import numpy as np
	
f = Flow().add(
    uses='jinahub+docker://MilvusIndexer',
    uses_with={
        'host': 'localhost',
        'port': 19530,
        'collection_name': 'collection_name',
        'distance': 'IP',
        'n_dim': 256,
    },
    install_requirements=True
)

with f:
    f.post('/index', inputs=[Document(embedding=np.random.rand(256)) for _ in range(3)])
```

### via source code

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
 uses='jinahub://MilvusIndexer',
    uses_with={
        'host': 'localhost',
        'port': 19530,
        'collection_name': 'collection_name',
        'distance': 'IP',
        'n_dim': 256,
    }
)

with f:
    f.post('/index', inputs=[Document(embedding=np.random.rand(256)) for _ in range(3)])
```



## CRUD Operations

You can perform CRUD operations (create, read, update and delete) using the respective endpoints:

- `/index`: Add new data to `Milvus`. 
- `/search`: Query the `Milvus` index (created in `/index`) with your Documents.
- `/update`: Update Documents in `Milvus`.
- `/delete`: Delete Documents in `Milvus`.


## Vector Search

The following example shows how to perform vector search using`f.post(on='/search', inputs=[Document(embedding=np.array([1,1]))])`.


```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
         uses='jinahub://MilvusIndexer',
         uses_with={'n_dim': 2},
     )

with f:
    f.post(
        on='/index',
        inputs=[
            Document(id='a', embedding=np.array([1, 3])),
            Document(id='b', embedding=np.array([1, 1])),
        ],
    )

    docs = f.post(
        on='/search',
        inputs=[Document(embedding=np.array([1, 1]))],
    )

# will print "The ID of the best match of [1,1] is: b"
print('The ID of the best match of [1,1] is: ', docs[0].matches[0].id)
```

### Using filtering

To leverage a filter with  `MilvusIndexer` you should first define the columns and the dimension of your embedding space, with `columns` and `n_dim` respectively.
For instance :


```python
from jina import Flow
f = Flow().add(
    uses='jinahub://MilvusIndexer',
    uses_with={
        'n_dim': 3,
        'distance': 'L2',
        'columns': [('price', 'float')],
    },
)
```

Then you can pass a filter as a parameters when searching for document:

```python
from docarray import Document, DocumentArray
import numpy as np
docs = DocumentArray(
    [
        Document(id=f'r{i}', embedding=np.random.rand(3), tags={'price': i})
        for i in range(50)
    ]
)

max_price = 3.0
filter = f'price <= {max_price}'

with f:
    f.index(docs)
    doc_query = DocumentArray([Document(embedding=np.random.rand(3))])
    f.search(doc_query, parameters={'filter': filter})
```

### Limit results

In some cases, you will want to limit the total number of retrieved results. `MilvusIndexer` uses the `limit` argument 
from the `match` function to set this limit. Note that when using `shards=N`, the `limit=K` is the number of retrieved results for **each shard** and total number of retrieved results is `N*K`. By default, `limits` is set to `20`. For more information about shards, please read [Jina Documentation](https://docs.jina.ai/fundamentals/flow/topology/#partition-data-by-using-shards)

```python
f =  Flow().add(
    uses='jinahub://MilvusIndexer',
    uses_with={'match_args': {'limit': 10}}
)
```

### Configure other search behaviors

You can use `match_args` argument to pass arguments to the `match` function as below. The match function will be called
during `/search` endpoint.

```python
f =  Flow().add(
     uses='jinahub://MilvusIndexer',
     uses_with={
         'match_args': {
             'metric': 'L2',
             'use_scipy': True
         }
     }
)
```

- For more details about overriding configurations, please refer to [this page](https://docs.jina.ai/fundamentals/executor/executor-in-flow/#special-executor-attributes).
- You can find more about the `match` function at [this page](https://docarray.jina.ai/api/docarray.array.mixins.match/#docarray.array.mixins.match.MatchMixin.match).

### Configure the search behaviors on-the-fly

**At search time**, you can also pass arguments to config the `match` function. This can be useful when users want to query with different arguments for different data requests. For instance, the following codes query with a custom `limit` in `parameters` and only retrieve the top 100 nearest neighbors. This will override existing `match_args` if defined during Executor initialization.

```python
with f:
    f.search(
        inputs=Document(text='hello'), 
        parameters={'limit': 100},
    )
```

For more information please refer to the docarray [documentation](https://docarray.jina.ai/advanced/document-store/milvus/#vector-search-with-filter).