#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

find . -type f -name "chkp_*" -exec $SCRIPT_DIR/single_checkpoint_handle.sh $1 {} \;

