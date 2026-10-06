# harness-web and the examples, with every optional service reachable.
#
# Two stages: Node builds the front end, Python runs it. The frontend build
# never reaches the runtime image, so node_modules stays out of the result.

# ---- stage 1: the React front end ------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build
# Copy the manifests alone first, so a source edit does not reinstall npm.
COPY harness-web/package.json harness-web/package-lock.json ./
RUN npm ci

COPY harness-web/index.html harness-web/vite.config.js ./
COPY harness-web/src ./src
COPY harness-web/public ./public
RUN npm run build


# ---- stage 2: the runtime ---------------------------------------------------
FROM python:3.12-slim AS runtime

# git: the knowledge recipes shallow-clone OKF bundles from git URLs.
# curl: the compose healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# Most MCP servers are launched over stdio as `npx -y @scope/server-...` or
# `uvx some-server`. Without these, the child process cannot start and the
# client sees nothing but "Connection closed", which says nothing about why.
# Copy node and npm in rather than apt's older build, and keep the version
# pinned to the one the frontend stage already uses.
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# uv and uvx, for Python MCP servers published to PyPI.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies from the manifest before the source, so editing an
# example does not reinstall the world.
COPY pyproject.toml README.md ./
COPY harnessx ./harnessx
# [all] is every integration; [dev] adds pytest, which the tests service needs.
RUN pip install -e ".[all,dev]"

COPY examples ./examples
COPY tests ./tests
COPY harness-web/harness_web ./harness-web/harness_web
COPY harness-web/run.py ./harness-web/run.py
COPY --from=frontend /build/dist ./harness-web/dist

# Examples import `examples.*`; harness_web.worker imports both.
ENV PYTHONPATH=/app:/app/harness-web

EXPOSE 8765

# 0.0.0.0, not localhost: the port has to be reachable from outside the
# container. run.py defaults to 127.0.0.1 for a local developer, which is right
# there and wrong here.
CMD ["python", "harness-web/run.py", "--host", "0.0.0.0", "--port", "8765"]
