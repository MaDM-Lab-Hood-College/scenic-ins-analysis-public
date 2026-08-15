#!/bin/bash
printf 'USERNAME=%s\nUSER_UID=%s\nGROUPNAME=%s\nGROUP_GID=%s\n' \
    "$USER" \
    "$(id -u "$USER")" \
    "$(id -gn "$USER" | sed 's/ /_/g')" \
    "$(id -g "$USER")" \
    > .devcontainer/.env
