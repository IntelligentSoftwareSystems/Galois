find $* -name '*.h' -o -name '*.cpp'  \
  | xargs grep --no-filename '#if' \
  | awk '{print $2;}' | sort | uniq
