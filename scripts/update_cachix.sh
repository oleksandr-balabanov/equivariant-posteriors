#!/usr/bin/env bash
nix build --json   | jq -r '.[].outputs | to_entries[].value'   | cachix push equivariant-posteriors
