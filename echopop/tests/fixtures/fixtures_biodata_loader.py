import os
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

HERE = Path(__file__).parent.absolute()
TEST_DATA_ROOT = HERE.parent / "test_data"
TEST_SQL_FILE = TEST_DATA_ROOT / "ingest" / "test_bio_data.sql"


@pytest.fixture(scope="session")
def postgres_container():
    """
    Session-scoped fixture to get database connection.

    - In GitHub Actions: Uses the postgres service from the workflow
    - Locally: Uses Testcontainers if Docker is available, skips if not
    """
    is_github_action = os.environ.get("GITHUB_ACTIONS")

    if is_github_action:
        # In GitHub Actions use the postgres service from workflow
        yield type(
            "obj",
            (object,),
            {
                "get_connection_url": lambda self: "postgresql+psycopg://test_user:postgres@localhost:5432/test",
                "get_container_host_ip": lambda self: "localhost",
                "get_exposed_port": lambda self, port: 5432,
            },
        )()
    else:
        # Local development
        try:
            from testcontainers.postgres import PostgresContainer

            container = PostgresContainer(
                image="postgres:16", username="test_user", password="postgres", dbname="test"
            )
            container.start()
            yield container
            container.stop()
        except Exception as e:
            # Docker not available - skip integration tests
            pytest.skip(f"Docker must be running for Testcontainers: {e}")


@pytest.fixture(scope="session")
def database_credentials(postgres_container):
    """
    Session-scoped fixture to:
    1. Connect to the PostgreSQL database (CI or local).
    2. Load 'test_bio_data.sql' into it.
    3. Yield the credentials dictionary in the format expected by load_biodata_db_views.

    Returns dict with keys: host, port, dbname, user, password, schema
    """

    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)

    creds = {
        "host": host,
        "port": port,
        "dbname": "test",
        "user": "test_user",
        "password": "postgres",
        "schema": "public",
    }

    db_url = (
        f"postgresql+psycopg://"
        f"{creds['user']}:{creds['password']}@"
        f"{creds['host']}:{creds['port']}/"
        f"{creds['dbname']}"
    )

    try:
        engine = create_engine(db_url)
        with engine.begin() as connection:
            with open(TEST_SQL_FILE) as f:
                sql_script = f.read()
                connection.execute(text(sql_script))
    except Exception as e:
        pytest.fail(f"Failed to load {TEST_SQL_FILE}: {e}")

    yield creds


@pytest.fixture
def biological_data():
    """Create sample biological data for testing."""
    # Create sample data for different sheets
    catch_data = pd.DataFrame(
        {
            "haul": [1, 2, 3, 4],
            "ship_id": [160, 160, 584, 584],
            "survey": [201906, 201906, 2019097, 2019097],
            "species_code": [22500, 22500, 22500, 30420],
            "weight_in_haul": [150.5, 200.3, 175.8, 90.2],
        }
    )

    length_data = pd.DataFrame(
        {
            "haul": [1, 1, 2, 3, 4],
            "ship_id": [160, 160, 160, 584, 584],
            "survey": [201906, 201906, 201906, 2019097, 2019097],
            "species_code": [22500, 22500, 22500, 22500, 30420],
            "sex": [1, 2, 1, 3, 2],
            "length": [45.2, 48.7, 42.5, 51.3, 38.9],
            "frequency": [10, 15, 8, 12, 5],
        }
    )

    specimen_data = pd.DataFrame(
        {
            "haul": [1, 2, 3, 4],
            "ship_id": [160, 160, 584, 584],
            "survey": [201906, 201906, 2019097, 2019097],
            "species_code": [22500, 22500, 22500, 30420],
            "sex": [1, 1, 2, 2],
            "length": [45.2, 42.5, 51.3, 38.9],
            "age": [3, 2, 4, 1],
        }
    )

    return {"catch": catch_data, "length": length_data, "specimen": specimen_data}


@pytest.fixture
def bio_excel_file(biological_data, tmp_path):
    """Create a temporary Excel file with biological data sheets."""
    file_path = tmp_path / "test_biodata.xlsx"

    with pd.ExcelWriter(file_path) as writer:
        biological_data["catch"].to_excel(writer, sheet_name="biodata_catch", index=False)
        biological_data["length"].to_excel(writer, sheet_name="biodata_length", index=False)
        biological_data["specimen"].to_excel(writer, sheet_name="biodata_specimen", index=False)

    return file_path


@pytest.fixture
def bio_sheet_map():
    """Create sheet mapping for biological data."""
    return {"catch": "biodata_catch", "length": "biodata_length", "specimen": "biodata_specimen"}


@pytest.fixture
def bio_column_map():
    """Create column mapping for biological data."""
    return {"frequency": "length_count", "haul": "haul_num", "weight_in_haul": "haul_weight"}


@pytest.fixture
def subset_dict():
    """Create subset dictionary for filtering biological data."""
    return {
        "ships": {160: {"survey": 201906}, 584: {"survey": 2019097, "haul_offset": 200}},
        "species_code": [22500],
    }


@pytest.fixture
def pg_subset_dict():
    """Create subset dictionary for filtering biological data."""
    return {
        "ships": {101: {"survey": 2024}},
        "species_code": [22500],
    }


@pytest.fixture
def bio_data_table_map():
    """Create table mapping for biological data in the database."""
    return {"catch": "echopop_catch", "specimen": "echopop_fish"}


@pytest.fixture
def column_name_map():
    """Create column mapping for biological data loaded from the database."""
    return {
        "haul": "haul_num",
        "weight_in_haul": "haul_weight",
        "species_id": "species_code",
    }


@pytest.fixture
def label_map():
    """Create label mapping dictionary for biological data."""
    return {"sex": {1: "male", 2: "female", 3: "unsexed"}}
