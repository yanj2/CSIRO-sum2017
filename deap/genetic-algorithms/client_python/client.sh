#!/bin/sh
cp -n ../library/bbcomp.dll      . 2>/dev/null || :
cp -n ../library/libbbcomp.so    . 2>/dev/null || :
cp -n ../library/libbbcomp.dylib . 2>/dev/null || :
version2="`python --version 2>&1 >/dev/null | grep 'Python 2.'`"
version3="`python --version 2>&1 >/dev/null | grep 'Python 3.'`"
if [ "A$version2" != "A" ];
then
	python client2.py
elif [ "A$version3" != "A" ];
then
	python client3.py
else
	echo "Failed to detect Python version."
fi
