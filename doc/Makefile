default: html

all: html latex test

check: test-noninteractive test

html:
	sphinx-build -E -b html . build/html

latex:
	sphinx-build -E -b latex . build/pdf

test:
	sphinx-build -b doctest . build/test

test-noninteractive:
	sphinx-build -b doctest . build/test-ni *.rst demos/inference/inference.rst

clean:
	rm -rf build

