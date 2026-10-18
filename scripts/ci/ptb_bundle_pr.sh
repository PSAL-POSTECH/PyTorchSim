#!/usr/bin/env bash
# Proposes the kernel bundle scripts/ci/bundle.py wrote (BUNDLE) to the compiler as BUNDLE_DIR on its
# main: one branch (BOT_BRANCH) force-pushed from main's tip, one open pull request on it. Nothing is
# proposed when the kernels are main's; bundle.json alone (the commit that wrote them) is no change.
# DRY_RUN=1 makes only read calls and prints the writes it would make.
set -euo pipefail

: "${PTB:=PSAL-POSTECH/PyTorchSim-Triton-Backend}"
: "${PTB_BASE:=main}"
: "${BOT_BRANCH:=bot/pts-bundle}"
: "${BUNDLE_DIR:=validation/pytorchsim}"
: "${DRY_RUN:=0}"
: "${PTB_TOKEN:?write token for $PTB (contents, pull requests)}"
: "${BUNDLE:?the directory scripts/ci/bundle.py wrote}"
: "${SOURCE_SHA:?the PyTorchSim commit that wrote it}"
MARKER='<!-- pts-bundle -->'
NAME='Wonhyuk Yang'
EMAIL='wonhyuk@postech.ac.kr'

api() {  # method path [json]
  local method=$1 path=$2 data=${3:-}
  if [ "$method" != GET ] && [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: $method $path ${data:0:300}" >&2; echo '{}'; return
  fi
  curl -fsSL -X "$method" -H "Authorization: Bearer $PTB_TOKEN" -H "Accept: application/vnd.github+json" \
    ${data:+-d "$data"} "https://api.github.com/repos/$PTB/$path"
}

BUNDLE=$(cd "$BUNDLE" && pwd)
OPEN=$(api GET "pulls?state=open&base=$PTB_BASE&per_page=100")
OURS=$(jq -r --arg b "$BOT_BRANCH" --arg o "${PTB%%/*}" \
  '[.[] | select(.head.ref == $b and .head.repo.owner.login == $o)][0].number // empty' <<<"$OPEN")

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 --branch "$PTB_BASE" "https://x-access-token:$PTB_TOKEN@github.com/$PTB.git" "$WORK/ptb"
cd "$WORK/ptb"
rm -rf "$BUNDLE_DIR"
mkdir -p "$(dirname "$BUNDLE_DIR")"
cp -r "$BUNDLE" "$BUNDLE_DIR"
git add -A "$BUNDLE_DIR"
if git diff --cached --quiet -- "$BUNDLE_DIR" ":!$BUNDLE_DIR/bundle.json"; then
  echo "the kernels are $PTB_BASE's: nothing to propose"
  if [ -n "$OURS" ]; then
    api POST "issues/$OURS/comments" "$(jq -nc --arg b "PyTorchSim ${SOURCE_SHA:0:7} builds the kernels $PTB_BASE already holds." '{body:$b}')" >/dev/null
    api PATCH "pulls/$OURS" '{"state":"closed"}' >/dev/null
  fi
  exit 0
fi

S=${SOURCE_SHA:0:7}
STAT=$(git diff --cached --shortstat -- "$BUNDLE_DIR")
STAT=${STAT# }
WORKLOADS=$(git diff --cached --name-only -- "$BUNDLE_DIR" | cut -d/ -f3 | grep -v '^bundle.json$' | sort -u)
N=$(wc -l <<<"$WORKLOADS")
[ "$N" -le 40 ] || WORKLOADS="$(head -40 <<<"$WORKLOADS")"$'\n'"... $((N - 40)) more"
GIT_AUTHOR_NAME=$NAME GIT_AUTHOR_EMAIL=$EMAIL GIT_COMMITTER_NAME=$NAME GIT_COMMITTER_EMAIL=$EMAIL \
  git commit -q -F - <<EOF
[Validation] The PyTorchSim kernels are PyTorchSim $S's

$BUNDLE_DIR is the bundle PyTorchSim $S builds for the validation workloads.
https://github.com/PSAL-POSTECH/PyTorchSim/commit/$SOURCE_SHA
EOF
git show --stat HEAD | tail -3
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: git push --force origin HEAD:refs/heads/$BOT_BRANCH"
else git push -q --force origin "HEAD:refs/heads/$BOT_BRANCH"; fi

TITLE="[Validation] PyTorchSim kernels: PyTorchSim $S"
BODY="Replaces \`$BUNDLE_DIR\` with the kernels PyTorchSim $S builds for the validation workloads ($STAT).

Workloads whose kernels changed:
\`\`\`
$WORKLOADS
\`\`\`

Opened by PyTorchSim's Triton route workflow. A newer PyTorchSim commit force-pushes \`$BOT_BRANCH\` and rewrites this pull request, so only the newest bundle is ever open.
$MARKER"
if [ -n "$OURS" ]; then
  echo "updating #$OURS"
  api PATCH "pulls/$OURS" "$(jq -nc --arg t "$TITLE" --arg b "$BODY" '{title:$t, body:$b}')" >/dev/null
else
  echo "opening a pull request"
  api POST "pulls" "$(jq -nc --arg t "$TITLE" --arg b "$BODY" --arg h "$BOT_BRANCH" --arg base "$PTB_BASE" \
    '{title:$t, body:$b, head:$h, base:$base}')" | jq -r '.html_url // empty'
fi
[ "$DRY_RUN" != 1 ] || printf '%s\n---\n%s\n' "$TITLE" "$BODY"
