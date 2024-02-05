from store import ExecutorPgVector, ExtendedPgVector, FullyAsyncPgVector


def get_vector_store(
    sync_connection_string: str,
    async_connection_string: str,
    embeddings,
    mode: str = "sync",
):
    if mode == "sync":
        return ExtendedPgVector(
            connection_string=sync_connection_string, embedding_function=embeddings
        )
    elif mode == "executor":
        return ExecutorPgVector(
            connection_string=sync_connection_string, embedding_function=embeddings
        )
    elif mode == "async":
        return FullyAsyncPgVector(
            connection_string=sync_connection_string,
            async_connection_string=async_connection_string,
            embedding_function=embeddings,
        )
    else:
        raise ValueError(
            "Invalid mode specified. Choose 'sync', 'executor', or 'async'."
        )


# Example usage
# pgvector_store = get_vector_store("postgresql+psycopg2://...", "postgresql+asyncpg://...", embeddings, mode="async")
