#!/bin/bash

MODE=$1
AVAIL_MODES="from_sc to_sc"
LOCAL=$HOME/workspace/vista/
REMOTE=tsunw@supercloud:/home/gridsan/tsunw/workspace/vista-integrate-new-api/

contains() {
    [[ " $AVAIL_MODES " =~ " $MODE " ]] && return 0 || return 1
}

if `contains $AVAIL_MODES $MODE`; then
    # double-check
    if [ $MODE == "from_sc" ]; then # sync from supercloud
        printf "Source: $REMOTE \nDestination: $LOCAL \n"
    else # upload to supercloud
        printf "Source: $LOCAL \nDestination: $REMOTE \n"
    fi
    read -p "Are you sure to proceed? [yes/no]: " proceed

    # run
    if [ ${proceed} == "yes" ]; then
        if [ $MODE == "from_sc" ]; then # sync from supercloud
            rsync -aP --exclude-from=$EXCLUDE_FROM --exclude='.git/' $REMOTE $LOCAL
        else # upload to supercloud
            rsync -aP --exclude-from=$EXCLUDE_FROM --exclude='.git/' $LOCAL $REMOTE
        fi
    else
        printf "Aborted!\n"
    fi
else
    printf "Invalid mode ${MODE}; only ${AVAIL_MODES} are supported!!\n"
fi