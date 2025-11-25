.PHONY: setup part1 part2 part3 part4 clean

setup:
	pip install -r requirements.txt

part1:
	cd src/part1_tree_manual
	python tree_manual.py