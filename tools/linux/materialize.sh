#!/system/bin/sh
# Before pack.sh: proot's hard links made real files again, from OUTSIDE the Linux (toybox, as the app):
#
#   adb push tools/linux/materialize.sh /data/local/tmp/
#   adb shell run-as xyz.tifu.geode sh /data/local/tmp/materialize.sh files/linux/rootfs
#
# Android does not let an app make hard links, so proot (--link2symlink) moves a linked file's data to a .l2s.* file
# and leaves symbolic links to it - links that name the phone's own path to the app's files, which mean nothing on
# another phone. Each such link becomes a copy of its data, and the .l2s.* files go.
set -e
cd "$1"
n=0
find . -type l | while read -r l; do
    case "$(readlink "$l")" in
        */.l2s.*)
            f=$(readlink -f "$l")
            cp -p "$f" "$l.materialized"
            mv -f "$l.materialized" "$l"
            n=$((n + 1))
            echo "$n $l"
            ;;
    esac
done | tail -1
find . -name '.l2s.*' -type f -exec rm -f {} +
echo "left: $(find . -name '.l2s.*' | wc -l) .l2s files"
