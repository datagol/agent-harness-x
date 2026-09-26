"""Portable, content-addressed artifacts; persisted state contains no host paths."""

from __future__ import annotations
import asyncio
import hashlib
from pathlib import Path
import tempfile
import os


class DirectoryArtifactStore:
    """Snapshot-owned artifacts, retained until PersistentMemory.delete_session()."""

    def __init__(self, root):
        self.root = Path(root)

    async def put_artifact(self, content: bytes) -> str:
        key = hashlib.sha256(content).hexdigest()

        def write():
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as output:
                temporary = output.name
                try:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                    os.replace(temporary, self.root / key)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        await asyncio.to_thread(write)
        return key

    async def get_artifact(self, key: str) -> bytes:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact key")
        content = await asyncio.to_thread((self.root / key).read_bytes)
        if hashlib.sha256(content).hexdigest() != key:
            raise ValueError("Artifact checksum mismatch")
        return content


def replace_paths(value, replacements):
    if isinstance(value, str):
        for old, new in sorted(
            replacements.items(), key=lambda x: len(x[0]), reverse=True
        ):
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {k: replace_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, replacements) for v in value]
    return value


async def capture(agent, state):
    store = getattr(agent, "_artifact_store", None)
    if store is None:
        return state
    paths = set(getattr(agent.memory, "_artifact_paths", set()))
    for ext in state.get("extensions", {}).values():
        if isinstance(ext, dict):
            for result in ext.get("results", {}).values():
                if isinstance(result, dict) and "path" in result:
                    paths.add(result["path"])
    replacements = dict(getattr(agent, "_artifact_uris", {}))
    for path in paths:
        if path.startswith("artifact://"):
            continue
        content = await asyncio.to_thread(Path(path).read_bytes)
        key = await store.put_artifact(content)
        replacements[path] = "artifact://" + key
    agent._artifact_uris = replacements
    return replace_paths(state, replacements)


async def materialize(agent, state):
    store = getattr(agent, "_artifact_store", None)
    if store is None:
        return state
    import json
    import re

    keys = set(re.findall(r"artifact://([a-f0-9]{64})", json.dumps(state)))
    replacements = {}
    if not keys:
        return state
    if getattr(agent, "_materialized_artifacts", None) is None:
        agent._materialized_artifacts = tempfile.TemporaryDirectory(prefix="harnessx-artifacts-")
    root = Path(agent._materialized_artifacts.name)
    for key in keys:
        path = root / key
        content = await store.get_artifact(key)
        if hashlib.sha256(content).hexdigest() != key:
            raise ValueError("Artifact checksum mismatch")
        await asyncio.to_thread(path.write_bytes, content)
        replacements["artifact://" + key] = str(path)
    agent._artifact_uris = {path: uri for uri, path in replacements.items()}
    agent.memory._artifact_paths.update(replacements.values())
    return replace_paths(state, replacements)


class S3ArtifactStore:
    def __init__(self, bucket, *, prefix="harness-x/", endpoint_url=None, client=None):
        self._owns_client = client is None
        self.bucket, self.prefix, self.endpoint_url, self._client = (
            bucket,
            prefix,
            endpoint_url,
            client,
        )

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    async def put_artifact(self, content):
        key = hashlib.sha256(content).hexdigest()
        await asyncio.to_thread(
            self._get_client().put_object,
            Bucket=self.bucket,
            Key=self.prefix + key,
            Body=content,
        )
        return key

    async def get_artifact(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact key")

        def read():
            body = self._get_client().get_object(
                Bucket=self.bucket, Key=self.prefix + key
            )["Body"]
            try:
                return body.read()
            finally:
                body.close()

        return await asyncio.to_thread(read)

    async def aclose(self):
        if self._owns_client and self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
