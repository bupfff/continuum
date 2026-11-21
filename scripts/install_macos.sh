#!/bin/sh
# Installs Continuum Viewer (Macs with Apple silicon) from the latest GitHub
# release into ~/Applications and opens it:
#
#   curl -fsSL https://raw.githubusercontent.com/bupfff/continuum/main/scripts/install_macos.sh | sh
#
# The download is checked against the SHA-256 digest GitHub publishes for
# it. Fetched this way the app is not marked as downloaded from the
# internet, so macOS opens it without the warning a browser download gets
# (the app is signed ad hoc, not notarized). `CONTINUUM_DIR` installs
# elsewhere; `CONTINUUM_NO_OPEN=1` skips opening it.
set -eu

repo="bupfff/continuum"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "The download is for Macs with Apple silicon. Elsewhere, build from source:"
    echo "  https://github.com/$repo#try-it"
    exit 1
fi
dir="${CONTINUUM_DIR:-$HOME/Applications}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "https://api.github.com/repos/$repo/releases/latest" -o "$tmp/release.json"
tag=$(plutil -extract tag_name raw -o - "$tmp/release.json")
url=$(plutil -extract assets.0.browser_download_url raw -o - "$tmp/release.json")
digest=$(plutil -extract assets.0.digest raw -o - "$tmp/release.json" 2>/dev/null || true)
case "$url" in
    *macos-arm64.zip) ;;
    *) echo "The latest release ($tag) has no macOS download."; exit 1 ;;
esac

echo "Downloading Continuum Viewer $tag…"
curl -fL --progress-bar "$url" -o "$tmp/viewer.zip"
if [ -n "$digest" ]; then
    actual="sha256:$(shasum -a 256 "$tmp/viewer.zip" | cut -d' ' -f1)"
    if [ "$actual" != "$digest" ]; then
        echo "The download does not match its published digest; nothing was installed."
        exit 1
    fi
fi

mkdir -p "$dir"
rm -rf "$dir/Continuum Viewer.app"
ditto -x -k "$tmp/viewer.zip" "$dir"
echo "Installed $dir/Continuum Viewer.app"
if [ "${CONTINUUM_NO_OPEN:-}" != 1 ]; then
    open "$dir/Continuum Viewer.app"
fi
