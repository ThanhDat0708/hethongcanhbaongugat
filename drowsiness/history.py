import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


@dataclass(frozen=True)
class HistoryEntry:
    timestamp: str
    category: str
    message: str


class EventHistory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL
                )
                """
            )
    def record(self, entry: HistoryEntry) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO events (timestamp, category, message) VALUES (?, ?, ?)",
                (entry.timestamp, entry.category, entry.message),
            )

    def recent(self, limit: int = 100) -> list[HistoryEntry]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT timestamp, category, message FROM events "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            HistoryEntry(timestamp, category, message)
            for timestamp, category, message in reversed(rows)
        ]

    def count(self, category: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM events WHERE category = ?", (category,)
            ).fetchone()
        if row is None:
            raise RuntimeError("Không thể đọc số lượng sự kiện trong lịch sử.")
        return int(row[0])

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()
