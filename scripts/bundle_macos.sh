#!/bin/bash
# Builds the viewer and wraps it in a macOS app bundle, zipped for a GitHub
# release: dist/continuum-viewer-<version>-macos-arm64.zip. The bundle is
# signed ad hoc (Apple silicon runs nothing unsigned), not notarized, so the
# first launch needs the steps in the README's "Try it".
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)"/\1/')
cargo build --release -p viewer
app="dist/Continuum Viewer.app"
rm -rf dist
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp target/release/viewer "$app/Contents/MacOS/viewer"
strip -x "$app/Contents/MacOS/viewer"
cp apps/viewer/macos/Continuum.icns "$app/Contents/Resources/"
sed "s/@VERSION@/$version/g" apps/viewer/macos/Info.plist > "$app/Contents/Info.plist"
codesign --force --deep --sign - "$app"
(cd dist && ditto -c -k --keepParent "Continuum Viewer.app" "continuum-viewer-$version-macos-arm64.zip")
ls -l dist
