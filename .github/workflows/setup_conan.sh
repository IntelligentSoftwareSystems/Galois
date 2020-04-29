#!/bin/bash
conan profile new --detect --force default
case $(uname) in
  Darwin*)
    ;;
  *) conan profile update settings.compiler.libcxx=libstdc++11 default
    ;;
esac
conan remote add kmaragon https://api.bintray.com/conan/kmaragon/conan
