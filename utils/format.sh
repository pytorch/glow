#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

if ! command -v clang-format; then
    echo "ERROR: can't find clang-format in your path."
    exit 1
fi

find lib tests/unittests/ tools/ include examples \
     -name \*.h -print0 \
     -o -name \*.cpp -print0 \
    | xargs -0 -P8 -n1 clang-format -i
