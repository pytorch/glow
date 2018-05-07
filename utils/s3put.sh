#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $(basename $0) model-directory"
    exit 1
fi

MODEL=$1

# You'll need an access key and secret key for this to work.  Search the
# FB-internal Glow group for `s3cmd` to find it.
s3cmd put --recursive --acl-public $MODEL s3://fb-glow-assets/models/
