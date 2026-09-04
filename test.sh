#!/usr/bin/bash
set -eou pipefail

poetry run mypy

poetry run ruff check

poetry run pytest