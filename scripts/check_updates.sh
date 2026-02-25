#!/bin/bash

check_updates() {
    is_up_to_date=true

    branch="$(git rev-parse --abbrev-ref HEAD)"

    # Update remote-tracking refs (no merge)
    git fetch origin "$branch" --quiet

    local_sha="$(git rev-parse "$branch")"
    remote_sha="$(git rev-parse "origin/$branch")"

    if [[ "$local_sha" != "$remote_sha" ]]; then
    is_up_to_date=false
    fi

    echo "$is_up_to_date"
}

# check_updates