"""Request-scoped memory partitions for trusted MCP gateways.

The feature is disabled unless ``OMBRE_PARTITION_GATEWAY_SECRET`` is set.
When enabled, the HTTP middleware authenticates a gateway-supplied owner and
the bucket-manager view used by MCP tools enforces that owner's tag. Dashboard
and management routes continue to use the unfiltered BucketManager instance.
"""

from __future__ import annotations

import contextvars
import hmac
import json
import os
import re
from typing import Any, Callable


OWNER_HEADER = b"x-ombre-partition-owner"
SECRET_HEADER = b"x-ombre-partition-key"
OWNER_TAG_PREFIX = "companion_"
_OWNER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_DEFAULT_OWNERS = frozenset({"gege", "claude", "deepseek"})
_owner_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ombre_partition_owner",
    default="",
)


def current_partition_owner() -> str:
    """Return the authenticated owner for the current MCP request."""

    return _owner_var.get()


def owner_tag(owner: str | None = None) -> str:
    normalized = str(owner if owner is not None else current_partition_owner()).strip().lower()
    return f"{OWNER_TAG_PREFIX}{normalized}" if normalized else ""


def _allowed_owners(raw: str | None = None) -> frozenset[str]:
    configured = str(
        raw if raw is not None else os.getenv("OMBRE_PARTITION_OWNERS", "")
    ).strip()
    if not configured:
        return _DEFAULT_OWNERS
    return frozenset(
        owner.strip().lower()
        for owner in configured.split(",")
        if _OWNER_PATTERN.fullmatch(owner.strip().lower())
    )


class MCPPartitionMiddleware:
    """Authenticate partition headers and bind the owner to this ASGI call."""

    def __init__(
        self,
        app: Any,
        *,
        path_matcher: Callable[[object], bool],
        secret: str | None = None,
        allowed_owners: str | None = None,
    ) -> None:
        self.app = app
        self.path_matcher = path_matcher
        self.secret = str(
            secret
            if secret is not None
            else os.getenv("OMBRE_PARTITION_GATEWAY_SECRET", "")
        ).strip()
        self.allowed_owners = _allowed_owners(allowed_owners)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if (
            not self.secret
            or scope.get("type") != "http"
            or not self.path_matcher(scope.get("path", ""))
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            bytes(key).lower(): bytes(value)
            for key, value in scope.get("headers", [])
        }
        supplied_secret = headers.get(SECRET_HEADER, b"").decode(
            "utf-8", errors="replace"
        )
        owner = headers.get(OWNER_HEADER, b"").decode(
            "utf-8", errors="replace"
        ).strip().lower()
        authorized = (
            _OWNER_PATTERN.fullmatch(owner) is not None
            and owner in self.allowed_owners
            and hmac.compare_digest(supplied_secret, self.secret)
        )
        if not authorized:
            body = json.dumps(
                {"error": "invalid_memory_partition"},
                separators=(",", ":"),
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        token = _owner_var.set(owner)
        try:
            await self.app(scope, receive, send)
        finally:
            _owner_var.reset(token)


class PartitionedBucketManager:
    """BucketManager view that confines MCP tools to the request owner."""

    def __init__(self, bucket_manager: Any) -> None:
        self._raw = bucket_manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    @staticmethod
    def _metadata(bucket: dict | None) -> dict:
        metadata = (bucket or {}).get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _visible(self, bucket: dict | None) -> bool:
        tag = owner_tag()
        if not tag:
            return bucket is not None
        tags = self._metadata(bucket).get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return bucket is not None and tag in {str(item).strip() for item in tags}

    @staticmethod
    def _partitioned_tags(tags: Any) -> Any:
        tag = owner_tag()
        if not tag:
            return tags
        if isinstance(tags, str):
            values = [tags]
        elif isinstance(tags, (list, tuple, set)):
            values = list(tags)
        else:
            values = []
        safe = [
            str(value).strip()
            for value in values
            if str(value).strip()
            and not str(value).strip().lower().startswith(OWNER_TAG_PREFIX)
        ]
        return list(dict.fromkeys([tag, *safe]))

    async def create(self, *args: Any, **kwargs: Any) -> str:
        if owner_tag():
            if len(args) >= 2:
                mutable = list(args)
                mutable[1] = self._partitioned_tags(mutable[1])
                args = tuple(mutable)
            else:
                kwargs["tags"] = self._partitioned_tags(kwargs.get("tags"))
        return await self._raw.create(*args, **kwargs)

    async def get(self, bucket_id: str) -> dict | None:
        bucket = await self._raw.get(bucket_id)
        return bucket if self._visible(bucket) else None

    async def get_including_archive(self, bucket_id: str) -> dict | None:
        getter = getattr(self._raw, "get_including_archive", self._raw.get)
        bucket = await getter(bucket_id)
        return bucket if self._visible(bucket) else None

    def find_exact_content(
        self,
        content: str,
        domain_filter: list[str] | None = None,
    ) -> dict | None:
        bucket = self._raw.find_exact_content(content, domain_filter)
        return bucket if self._visible(bucket) else None

    async def list_all(self, include_archive: bool = False) -> list[dict]:
        buckets = await self._raw.list_all(include_archive=include_archive)
        return [bucket for bucket in buckets if self._visible(bucket)]

    async def search(self, query: str, limit: int | None = None, **kwargs: Any) -> list[dict]:
        requested = limit or getattr(self._raw, "max_results", 10)
        if not owner_tag():
            return await self._raw.search(query, limit=limit, **kwargs)
        all_buckets = await self._raw.list_all(
            include_archive=bool(kwargs.get("include_archive", False))
        )
        expanded_limit = max(requested, len(all_buckets), 1)
        results = await self._raw.search(query, limit=expanded_limit, **kwargs)
        return [bucket for bucket in results if self._visible(bucket)][:requested]

    async def update(self, bucket_id: str, **kwargs: Any) -> bool:
        if owner_tag() and not self._visible(await self._raw.get(bucket_id)):
            return False
        if "tags" in kwargs:
            kwargs["tags"] = self._partitioned_tags(kwargs["tags"])
        return await self._raw.update(bucket_id, **kwargs)

    async def _update_locked(self, bucket_id: str, **kwargs: Any) -> bool:
        if owner_tag() and not self._visible(await self._raw.get(bucket_id)):
            return False
        if "tags" in kwargs:
            kwargs["tags"] = self._partitioned_tags(kwargs["tags"])
        return await self._raw._update_locked(bucket_id, **kwargs)

    async def update_content_fragment(self, bucket_id: str, **kwargs: Any) -> dict:
        if owner_tag() and not self._visible(await self._raw.get(bucket_id)):
            return {"ok": False, "error": "not_found"}
        if "tags" in kwargs:
            kwargs["tags"] = self._partitioned_tags(kwargs["tags"])
        return await self._raw.update_content_fragment(bucket_id, **kwargs)

    async def delete(self, bucket_id: str) -> bool:
        if owner_tag() and not self._visible(await self.get_including_archive(bucket_id)):
            return False
        return await self._raw.delete(bucket_id)

    async def archive(self, bucket_id: str) -> bool:
        if owner_tag() and not self._visible(await self.get_including_archive(bucket_id)):
            return False
        return await self._raw.archive(bucket_id)

    async def restore_archived(self, bucket_id: str) -> dict:
        if owner_tag() and not self._visible(await self.get_including_archive(bucket_id)):
            return {"ok": False, "error": "not_found"}
        return await self._raw.restore_archived(bucket_id)

    async def hard_delete_test_bucket(self, bucket_id: str, *, reason: str = "") -> dict:
        if owner_tag() and not self._visible(await self.get_including_archive(bucket_id)):
            return {"ok": False, "error": "not_found"}
        return await self._raw.hard_delete_test_bucket(bucket_id, reason=reason)

    async def set_anchor(self, bucket_id: str, value: bool) -> dict:
        if owner_tag() and not self._visible(await self._raw.get(bucket_id)):
            return {"ok": False, "error": "bucket not found", "count": 0, "limit": 24}
        result = await self._raw.set_anchor(bucket_id, value)
        if owner_tag():
            anchors = await self.list_anchors()
            result = {**result, "count": len(anchors)}
        return result

    async def list_anchors(self) -> list[dict]:
        buckets = await self.list_all(include_archive=False)
        anchors = [
            bucket
            for bucket in buckets
            if self._metadata(bucket).get("anchor")
        ]
        anchors.sort(key=lambda bucket: self._metadata(bucket).get("created", ""))
        return anchors

    async def count_anchors(self) -> int:
        return len(await self.list_anchors())

    async def touch_many(self, bucket_ids: list, ripple: bool = False) -> None:
        if not owner_tag():
            await self._raw.touch_many(bucket_ids, ripple=ripple)
            return
        visible_ids = []
        for bucket_id in bucket_ids or []:
            if self._visible(await self._raw.get(bucket_id)):
                visible_ids.append(bucket_id)
        if visible_ids:
            await self._raw.touch_many(visible_ids, ripple=ripple)

    async def get_triggered_feels(self, source_bucket_id: str) -> list[dict]:
        if owner_tag() and not self._visible(await self._raw.get(source_bucket_id)):
            return []
        results = await self._raw.get_triggered_feels(source_bucket_id)
        visible = []
        for item in results:
            bucket = await self._raw.get(str(item.get("id") or ""))
            if self._visible(bucket):
                visible.append(item)
        return visible

    async def get_stats(self) -> dict:
        if not owner_tag():
            return await self._raw.get_stats()
        buckets = await self.list_all(include_archive=True)
        stats: dict[str, Any] = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "plan_count": 0,
            "letter_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }
        type_keys = {
            "permanent": "permanent_count",
            "dynamic": "dynamic_count",
            "archived": "archive_count",
            "feel": "feel_count",
            "plan": "plan_count",
            "letter": "letter_count",
        }
        for bucket in buckets:
            metadata = self._metadata(bucket)
            bucket_type = str(metadata.get("type") or "dynamic").lower()
            if metadata.get("deleted_at"):
                bucket_type = "archived"
            key = type_keys.get(bucket_type, "dynamic_count")
            stats[key] += 1
            stats["total_size_kb"] += len(
                str(bucket.get("content") or "").encode("utf-8")
            ) / 1024
            domains = metadata.get("domain") or []
            if isinstance(domains, str):
                domains = [domains]
            for domain in domains:
                normalized = str(domain).strip()
                if normalized:
                    stats["domains"][normalized] = stats["domains"].get(normalized, 0) + 1
        stats["total_size_kb"] = round(stats["total_size_kb"], 2)
        return stats

