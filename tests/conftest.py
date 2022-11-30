import os
import time

import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture(scope='module')
def docker_compose():
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d"
    )

    _wait_for_milvus()

    yield
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . down "
        f"--remove-orphans"
    )


def restart_milvus():
    os.system(f"docker-compose -f {compose_yml} --project-directory . down")
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d"
    )
    _wait_for_milvus(restart_on_failure=False)


def _wait_for_milvus(restart_on_failure=True):
    from pymilvus import connections, has_collection
    from pymilvus.exceptions import MilvusUnavailableException, MilvusException

    milvus_conn_alias = f'pytest_localhost_19530'
    try:
        connections.connect(alias=milvus_conn_alias, host='localhost', port=19530)
        milvus_ready = False
        while not milvus_ready:
            try:
                has_collection('ping', using=milvus_conn_alias)
                milvus_ready = True
            except MilvusUnavailableException:
                # Milvus is not ready yet, just wait
                time.sleep(0.5)
    except MilvusException as e:
        if e.code == 1 and restart_on_failure:
            # something went wrong with the docker container, restart and retry once
            restart_milvus()
        else:
            raise e
