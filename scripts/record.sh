# Source in Bash. BATCH is shared; STAGE is distinct (common/S00/S01/S05).
# Each label atomically claims a fresh directory, including under concurrent calls.
record() {
  local label="${1:?record requires a label}"
  shift
  local component
  for component in "${BATCH:?set BATCH}" "${STAGE:?set STAGE}" "$label"; do
    if [[ ! "$component" =~ ^[a-zA-Z0-9_.-]+$ || "$component" == '.' || "$component" == '..' ]]; then
      printf 'Invalid evidence path component: %s\n' "$component" >&2
      return 64
    fi
  done
  if (( $# == 0 )); then
    printf 'record requires a command\n' >&2
    return 64
  fi
  local parent="logs/$BATCH/$STAGE" base="logs/$BATCH/$STAGE/$label" rc
  mkdir -p "$parent" || return 73
  if ! mkdir "$base"; then
    printf 'Refusing to overwrite evidence: %s\n' "$base" >&2
    return 73
  fi
  date -u +%FT%TZ > "$base/started" || return 74
  printf '%q ' "$@" > "$base/argv" || return 74
  printf '\n' >> "$base/argv" || return 74
  # An explicit conditional preserves exit evidence when the caller uses set -e.
  if "$@" > "$base/output.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$base/exitcode" || return 74
  date -u +%FT%TZ > "$base/ended" || return 74
  return "$rc"
}
