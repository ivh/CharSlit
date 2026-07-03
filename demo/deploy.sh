#!/bin/sh
# Rebuild the demo and force-push it to the gh-pages branch.
# Site: https://ivh.github.io/CharSlit/
set -e
cd "$(dirname "$0")/.."

./demo/build.sh
uv run python demo/make_manifest.py

REMOTE=$(git remote get-url origin)
SITE=$(mktemp -d)
trap 'rm -rf "$SITE"' EXIT

cd demo
cp index.html style.css app.js worker.js io.js \
   slitdec.mjs slitdec.wasm swaths.json "$SITE/"
touch "$SITE/.nojekyll"
mkdir "$SITE/data"
uv run python -c "
import json, shutil
for s in json.load(open('swaths.json'))['swaths']:
    shutil.copy('../' + s['url'], '$SITE/' + s['url'])
"

cd "$SITE"
git init -q -b gh-pages
git add -A
git commit -q -m "Deploy slitdec wasm demo"
git push -f "$REMOTE" gh-pages:gh-pages
echo "deployed: https://ivh.github.io/CharSlit/ (allow a minute for Pages to rebuild)"
