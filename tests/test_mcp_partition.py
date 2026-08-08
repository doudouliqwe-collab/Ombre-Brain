from __future__ import annotations

import pytest

from ombrebrain.security import mcp_partition
from ombrebrain.security.mcp_partition import (
    MCPPartitionMiddleware,
    PartitionedBucketManager,
    current_partition_owner,
)


def _collect(messages):
    async def send(message):
        messages.append(message)

    return send


def _bucket(bucket_id: str, owner: str, *, content: str = "memory") -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "type": "dynamic",
            "tags": [f"companion_{owner}"],
            "domain": ["test"],
        },
    }


class FakeBucketManager:
    max_results = 10

    def __init__(self) -> None:
        self.buckets = {
            "g1": _bucket("g1", "gege", content="哥哥的记忆"),
            "c1": _bucket("c1", "claude", content="Claude 的记忆"),
        }
        self.last_create = None
        self.last_update = None
        self.last_search_limit = None
        self.touched = []
        self.destructive_calls = []

    async def create(self, *args, **kwargs):
        self.last_create = (args, kwargs)
        return "new"

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def get_including_archive(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def list_all(self, include_archive=False):
        return list(self.buckets.values())

    async def search(self, query, limit=None, **kwargs):
        self.last_search_limit = limit
        return list(self.buckets.values())[:limit]

    async def update(self, bucket_id, **kwargs):
        self.last_update = (bucket_id, kwargs)
        return True

    async def update_content_fragment(self, bucket_id, **kwargs):
        self.destructive_calls.append(("fragment", bucket_id))
        return {"ok": True}

    async def delete(self, bucket_id):
        self.destructive_calls.append(("delete", bucket_id))
        return True

    async def restore_archived(self, bucket_id):
        self.destructive_calls.append(("restore", bucket_id))
        return {"ok": True}

    async def hard_delete_test_bucket(self, bucket_id, *, reason=""):
        self.destructive_calls.append(("hard_delete", bucket_id))
        return {"ok": True}

    async def set_anchor(self, bucket_id, value):
        self.destructive_calls.append(("anchor", bucket_id))
        return {"ok": True, "count": 1, "limit": 24}

    async def touch_many(self, bucket_ids, ripple=False):
        self.touched = list(bucket_ids)

    async def get_stats(self):
        return {"unpartitioned": True}


@pytest.mark.asyncio
async def test_partition_middleware_rejects_missing_headers():
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    middleware = MCPPartitionMiddleware(
        app,
        path_matcher=lambda path: path == "/mcp",
        secret="shared-secret",
    )
    messages = []

    await middleware(
        {"type": "http", "path": "/mcp", "headers": []},
        lambda: None,
        _collect(messages),
    )

    assert called is False
    assert messages[0]["status"] == 403
    assert b"invalid_memory_partition" in messages[1]["body"]


@pytest.mark.asyncio
async def test_partition_middleware_binds_and_resets_owner():
    observed = []

    async def app(scope, receive, send):
        observed.append(current_partition_owner())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = MCPPartitionMiddleware(
        app,
        path_matcher=lambda path: path == "/mcp",
        secret="shared-secret",
    )
    messages = []

    await middleware(
        {
            "type": "http",
            "path": "/mcp",
            "headers": [
                (b"x-ombre-partition-owner", b"gege"),
                (b"x-ombre-partition-key", b"shared-secret"),
            ],
        },
        lambda: None,
        _collect(messages),
    )

    assert observed == ["gege"]
    assert messages[0]["status"] == 204
    assert current_partition_owner() == ""


@pytest.mark.asyncio
async def test_partitioned_manager_filters_reads_and_confines_writes():
    raw = FakeBucketManager()
    manager = PartitionedBucketManager(raw)
    token = mcp_partition._owner_var.set("gege")
    try:
        assert [bucket["id"] for bucket in await manager.list_all()] == ["g1"]
        assert await manager.get("c1") is None
        assert await manager.update("c1", content="blocked") is False

        assert await manager.update(
            "g1",
            tags=["companion_claude", "private"],
        ) is True
        assert raw.last_update == (
            "g1",
            {"tags": ["companion_gege", "private"]},
        )

        await manager.create(content="new", tags=["companion_deepseek", "note"])
        assert raw.last_create[1]["tags"] == ["companion_gege", "note"]

        results = await manager.search("memory", limit=1)
        assert [bucket["id"] for bucket in results] == ["g1"]
        assert raw.last_search_limit == 2

        await manager.touch_many(["g1", "c1"])
        assert raw.touched == ["g1"]

        assert await manager.update_content_fragment("c1", old_str="x", new_str="y") == {
            "ok": False,
            "error": "not_found",
        }
        assert await manager.delete("c1") is False
        assert await manager.restore_archived("c1") == {
            "ok": False,
            "error": "not_found",
        }
        assert await manager.hard_delete_test_bucket("c1", reason="test") == {
            "ok": False,
            "error": "not_found",
        }
        assert (await manager.set_anchor("c1", True))["ok"] is False
        assert raw.destructive_calls == []
    finally:
        mcp_partition._owner_var.reset(token)


@pytest.mark.asyncio
async def test_partition_feature_is_backward_compatible_without_owner_context():
    raw = FakeBucketManager()
    manager = PartitionedBucketManager(raw)

    assert len(await manager.list_all()) == 2
    assert await manager.get("c1") is not None
    assert await manager.get_stats() == {"unpartitioned": True}

