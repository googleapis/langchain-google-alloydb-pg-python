import asyncio
import os

from vertexai.preview import reasoning_engines

from langchain_google_alloydb_pg import AlloyDBEngine

PROJECT_ID = os.getenv("PROJECT_ID") or "my-project-id"
REGION = os.getenv("REGION") or "us-central1"
CLUSTER = os.getenv("CLUSTER") or "my-alloy-db"
INSTANCE = os.getenv("INSTANCE") or "my-primary"
DATABASE = os.getenv("DATABASE") or "my_database"
TABLE_NAME = os.getenv("TABLE_NAME") or "my_test_table"
CHAT_TABLE_NAME = os.getenv("CHAT_TABLE_NAME") or "my_chat_table"
USER = os.getenv("DB_USER") or "postgres"
PASSWORD = os.getenv("DB_PASSWORD") or "password"
TEST_NAME = os.getenv("DISPLAY_NAME")


async def delete_databases():
    engine = await AlloyDBEngine.afrom_instance(
        PROJECT_ID,
        REGION,
        CLUSTER,
        INSTANCE,
        database="postgres",
        user="postgres",
        password=PASSWORD,
    )

    await engine._aexecute_outside_tx(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    await engine._aexecute_outside_tx(f"DROP TABLE IF EXISTS {CHAT_TABLE_NAME}")


def delete_engines():
    apps = reasoning_engines.ReasoningEngine.list(
        filter=f'display_name="{TEST_NAME}"'
    )
    for app in apps:
        app.delete()


async def main():
    await delete_databases()
    delete_engines()


asyncio.run(main())
