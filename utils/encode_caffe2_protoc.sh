#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $(basename $0) pbtxt"
    exit 1
fi

PBTXT="$1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LIB_DIR="$SCRIPT_DIR/../lib/Importer"

protoc --encode=caffe2.NetDef -I "$LIB_DIR" "$LIB_DIR/caffe.proto" < "$PBTXT"
