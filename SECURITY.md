# Security policy

## Supported versions

The latest release on PyPI receives security fixes. Older releases do not.

## Reporting a vulnerability

Do not open a public issue for a security problem.

Use GitHub's private vulnerability reporting for this repository:
https://github.com/datagol/agent-harness-x/security/advisories/new

Include the version, a description of the impact, and steps to reproduce.
You will get an acknowledgement within five working days. Fixes are released
as a new PyPI version and noted in `CHANGELOG.md` once a fix is available.

## Scope notes

HarnessX runs model-chosen tool calls. The permission system, sandbox tiers,
and approval flow are security boundaries; a way to bypass them is in scope.
The examples are demonstrations for trusted local use and are documented as
such; reports about running them exposed to the network are out of scope.
