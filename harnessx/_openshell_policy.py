"""OpenShell sandbox policies from ``allow=`` and ``secrets=``, and from OpenShell's own YAML.

OpenShell decides network access per program: a rule lists hosts and the
executables that may reach them, and it also covers whatever those executables
start. Every tool command runs under ``sh -c``, so a rule that lists the shells
covers every command the agent runs, whatever program in it connects.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

# Real paths of the shells ``sh -c`` resolves to on common images (dash on
# Debian and Ubuntu, bash elsewhere, busybox on Alpine). OpenShell matches the
# path the kernel reports, never a symlink such as /bin/sh.
SHELL_BINARIES = (
    "/usr/bin/dash", "/usr/bin/bash", "/bin/dash", "/bin/bash", "/bin/busybox", "/usr/bin/busybox",
)

# Hosts a service needs besides the one people name.
COMPANION_HOSTS: dict[str, tuple[str, ...]] = {
    "github.com": ("api.github.com", "codeload.github.com", "objects.githubusercontent.com",
                   "raw.githubusercontent.com"),
    "pypi.org": ("files.pythonhosted.org",),
    "huggingface.co": ("cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"),
}

# Where well-known secrets are used, so ``secrets=["GITHUB_TOKEN"]`` needs no hosts.
SECRET_HOSTS: dict[str, tuple[str, ...]] = {
    "GITHUB_TOKEN": ("github.com", "api.github.com"),
    "GH_TOKEN": ("github.com", "api.github.com"),
    "HF_TOKEN": ("huggingface.co",),
    "NPM_TOKEN": ("registry.npmjs.org",),
    "ANTHROPIC_API_KEY": ("api.anthropic.com",),
    "OPENAI_API_KEY": ("api.openai.com",),
}

DEFAULT_READ_ONLY = ("/usr", "/lib", "/etc", "/proc", "/dev/urandom", "/var/log", "/bin", "/sbin", "/opt")
DEFAULT_READ_WRITE = ("/tmp", "/dev/null")

_HOST = re.compile(r"^(\*\*?\.)?[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*$")

_ENUMS = {
    "tls": {"": "NETWORK_TLS_MODE_UNSPECIFIED", "skip": "NETWORK_TLS_MODE_SKIP"},
    "enforcement": {"": "NETWORK_ENFORCEMENT_MODE_UNSPECIFIED", "enforce": "NETWORK_ENFORCEMENT_MODE_ENFORCE",
                    "audit": "NETWORK_ENFORCEMENT_MODE_AUDIT"},
    "access": {"": "NETWORK_ACCESS_PRESET_UNSPECIFIED", "read-only": "NETWORK_ACCESS_PRESET_READ_ONLY",
               "read-write": "NETWORK_ACCESS_PRESET_READ_WRITE", "full": "NETWORK_ACCESS_PRESET_FULL"},
}


def parse_host(entry: str) -> tuple[str, int]:
    """``"github.com"`` or ``"example.com:8443"`` as (host, port); port 443 by default."""
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("allow entries must be host names such as 'github.com' or 'example.com:8443'")
    host, _, port = entry.strip().lower().partition(":")
    if "/" in host or not _HOST.match(host):
        raise ValueError(f"allow entry {entry!r} is not a host name (no scheme or path: 'github.com')")
    if port and not (port.isdigit() and 0 < int(port) < 65536):
        raise ValueError(f"allow entry {entry!r} has an invalid port")
    return host, int(port) if port else 443


def expand_hosts(entries: Sequence[str]) -> list[tuple[str, int]]:
    """Hosts with their companions, without duplicates, in a stable order."""
    seen: list[tuple[str, int]] = []
    for entry in entries:
        host, port = parse_host(entry)
        for name in (host, *COMPANION_HOSTS.get(host, ())):
            if (name, port) not in seen:
                seen.append((name, port))
    return seen


def secret_groups(secrets: Sequence[str] | Mapping[str, str | Sequence[str]]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Secrets grouped by the hosts they are for: [((ENV_VAR, ...), (host, ...)), ...].

    A list names well-known variables; a mapping gives each variable its
    host or hosts. Secrets for the same hosts share one OpenShell provider,
    because an endpoint is bound to exactly one.
    """
    if isinstance(secrets, str):
        raise TypeError("secrets must be a list of environment variable names or a mapping of name to hosts")
    pairs: dict[str, tuple[str, ...]] = {}
    if isinstance(secrets, Mapping):
        for name, hosts in secrets.items():
            pairs[name] = (hosts,) if isinstance(hosts, str) else tuple(hosts)
    else:
        for name in secrets:
            if name not in SECRET_HOSTS:
                raise ValueError(
                    f"HarnessX does not know where {name!r} is used; say which host: "
                    f"secrets={{{name!r}: 'api.example.com'}}"
                )
            pairs[name] = SECRET_HOSTS[name]
    groups: dict[tuple[str, ...], list[str]] = {}
    for name, hosts in pairs.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"{name!r} is not an environment variable name")
        if not hosts:
            raise ValueError(f"secret {name!r} has no host")
        key = tuple(sorted({parse_host(host)[0] for host in hosts}))
        groups.setdefault(key, []).append(name)
    claimed: dict[str, tuple[str, ...]] = {}
    for hosts in groups:
        for host in hosts:
            if host in claimed:
                raise ValueError(
                    f"{host} is named by secrets with different host lists ({', '.join(claimed[host])} and "
                    f"{', '.join(hosts)}); give every secret for {host} the same hosts"
                )
            claimed[host] = hosts
    return [(tuple(sorted(names)), hosts) for hosts, names in groups.items()]


def _yaml_to_proto_dict(document: Mapping[str, Any]) -> dict[str, Any]:
    """OpenShell's policy YAML in the field names and enum spellings of its proto."""
    converted = dict(document)
    if "filesystem_policy" in converted:
        converted["filesystem"] = converted.pop("filesystem_policy")
    rules = {}
    for key, rule in (converted.get("network_policies") or {}).items():
        rule = dict(rule)
        rule.setdefault("name", key)
        rule["binaries"] = [{"path": b} if isinstance(b, str) else b for b in rule.get("binaries") or []]
        endpoints = []
        for endpoint in rule.get("endpoints") or []:
            endpoint = dict(endpoint)
            for field_name, names in _ENUMS.items():
                if field_name in endpoint and isinstance(endpoint[field_name], str):
                    value = endpoint[field_name].lower()
                    if value not in names:
                        raise ValueError(f"network rule {key!r}: unknown {field_name} {endpoint[field_name]!r}")
                    endpoint[field_name] = names[value]
            for list_name in ("rules", "deny_rules"):
                for entry in endpoint.get(list_name) or []:
                    target = entry.get("allow", entry) if list_name == "rules" else entry
                    query = target.get("query") if isinstance(target, dict) else None
                    if isinstance(query, dict):
                        target["query"] = {
                            name: {"glob": m} if isinstance(m, str) else {"any": list(m)} if isinstance(m, list) else m
                            for name, m in query.items()
                        }
            endpoints.append(endpoint)
        rule["endpoints"] = endpoints
        rules[key] = rule
    if rules:
        converted["network_policies"] = rules
    return converted


def load_policy(source: Any) -> Any:
    """A ``SandboxPolicy`` from a YAML file path, YAML text, a mapping, or a policy message."""
    import yaml
    from google.protobuf import json_format
    from openshell._proto import sandbox_pb2

    if isinstance(source, sandbox_pb2.SandboxPolicy):
        policy = sandbox_pb2.SandboxPolicy()
        policy.CopyFrom(source)
        return policy
    if isinstance(source, os.PathLike) or (isinstance(source, str) and "\n" not in source and os.path.isfile(source)):
        with open(source, encoding="utf-8") as stream:
            source = yaml.safe_load(stream)
    elif isinstance(source, str):
        source = yaml.safe_load(source)
    if not isinstance(source, Mapping):
        raise ValueError("policy must be a YAML file path, YAML text, or a mapping")
    policy = sandbox_pb2.SandboxPolicy()
    try:
        json_format.ParseDict(_yaml_to_proto_dict(source), policy)
    except json_format.ParseError as exc:
        raise ValueError(f"Not a valid OpenShell policy: {exc}") from None
    return policy


def build_policy(
    *, base: Any | None, allow: Sequence[tuple[str, int]], credentials: Mapping[str, str],
) -> Any:
    """The sandbox policy: ``base`` (or a restrictive default) plus a rule per allowed host.

    ``credentials`` maps a host to the provider whose secrets it may receive;
    such an endpoint is inspected (``protocol: rest``), which OpenShell
    requires before it rewrites a secret into a request.
    """
    from openshell._proto import sandbox_pb2

    policy = sandbox_pb2.SandboxPolicy()
    if base is not None:
        policy.CopyFrom(base)
    else:
        policy.filesystem.include_workdir = True
        policy.filesystem.read_only.extend(DEFAULT_READ_ONLY)
        policy.filesystem.read_write.extend(DEFAULT_READ_WRITE)
        policy.landlock.compatibility = "hard_requirement"
    policy.version = policy.version or 1
    for host, port in allow:
        name = "harnessx_" + re.sub(r"[^a-z0-9]+", "_", f"{host}_{port}").strip("_")
        rule = policy.network_policies[name]
        rule.name = name
        endpoint = rule.endpoints.add(host=host, ports=[port], enforcement=sandbox_pb2.NETWORK_ENFORCEMENT_MODE_ENFORCE)
        if host in credentials:
            endpoint.protocol = "rest"
            endpoint.access = sandbox_pb2.NETWORK_ACCESS_PRESET_FULL
            endpoint.credential_binding.provider = credentials[host]
        for binary in SHELL_BINARIES:
            rule.binaries.add(path=binary)
    return policy


_OCSF_DENIAL = re.compile(r"\bDENIED\b(?:\s+\S+\(\d+\)\s+->)?\s+(\S+?)(?:\s+\[|$)")


def denials_from_log(messages: Sequence[str]) -> list[str]:
    """Destinations OpenShell's audit log says it refused (``NET:OPEN ... DENIED bin -> host:port``)."""
    found: list[str] = []
    for message in messages:
        if not message.startswith(("NET:", "HTTP:")) or " DENIED " not in f" {message} ":
            continue
        match = _OCSF_DENIAL.search(message)
        target = match.group(1) if match else message
        if target not in found:
            found.append(target)
    return found
