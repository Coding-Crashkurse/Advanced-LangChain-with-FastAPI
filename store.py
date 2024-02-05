import asyncio
import enum
import logging
import time
from typing import Any, Callable, Optional

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import Session, sessionmaker

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE


class ExtendedPgVector(PGVector):

    def get_all_ids(self) -> list[str]:
        time.sleep(5)

        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:

        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(self.EmbeddingStore.custom_id.in_(ids))
                .all()
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
                if result.custom_id in ids
            ]


class ExecutorPgVector(ExtendedPgVector):

    async def get_all_ids(self) -> list[str]:
        await asyncio.sleep(5)
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        collection_only: bool = False,
        **kwargs: Any
    ) -> None:
        await run_in_executor(None, super().delete, ids, collection_only, **kwargs)


class FullyAsyncPgVector(PGVector):
    def __init__(
        self,
        connection_string: str,
        async_connection_string: str,
        embedding_function: Embeddings,
        embedding_length: Optional[int] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        engine_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            connection_string=connection_string,
            embedding_function=embedding_function,
            embedding_length=embedding_length,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            logger=logger,
            relevance_score_fn=relevance_score_fn,
            engine_args=engine_args,
        )
        self._async_engine = create_async_engine(
            async_connection_string, **self.engine_args
        )
        self._async_session = sessionmaker(
            bind=self._async_engine, class_=AsyncSession, expire_on_commit=False
        )

    async def _make_session(self) -> AsyncSession:
        """Create a context manager for the async session."""
        async with self._async_session() as session:
            yield session

    async def get_all_ids(self) -> list[str]:
        await asyncio.sleep(5)
        async with self._make_session() as session:
            results = await session.execute(select(self.EmbeddingStore.custom_id))
            return [
                result[0]
                for result in results.scalars().fetchall()
                if result[0] is not None
            ]

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        async with self._make_session() as session:
            results = await session.execute(
                select(self.EmbeddingStore).filter(
                    self.EmbeddingStore.custom_id.in_(ids)
                )
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results.scalars().fetchall()
                if result.custom_id in ids
            ]
