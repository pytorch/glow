#!/bin/bash

echo "["
tail -q -n +2 $@ | grep -h '"ph":"M"' | sort | uniq
tail -q -n +2 $@ | grep -h -v '"ph":"M"'
