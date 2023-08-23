#!/bin/bash
# For PR build only; find the longest common path prefix as the build and test subset

longest_common_prefix() {
    declare -a possible_prefix
    declare i=0

    path="${1%/}"
    while [ "$path" != "." ]; do
        if [[ -d $path && -f "$path/CMakeLists.txt" ]]; then
            possible_prefix[$i]="$path"
        fi
        i=$(($i + 1))
        path=$(dirname "$path");
    done

    lcp="."
    for prefix in "${possible_prefix[@]}"; do
        for path in $@; do
            if [ "${path#$prefix}" = "${path}" ]; then
                continue 2
            fi
        done
        lcp="$prefix"
        break
    done
    echo $lcp
}
base=$( \
    wget -q -O - "https://api.github.com/repos/$(echo ${CIRCLE_PULL_REQUEST:19} | sed "s/\/pull\//\/pulls\//")" \
    | sed -n -e "s/^.*IntelligentSoftwareSystems://p" \
    | sed -n -e "s/\".*$//p" \
)
longest_common_prefix $(git -c core.quotepath=false diff --name-only $base $CIRCLE_SHA1)