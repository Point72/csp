EXTRA_ARGS :=

#########
# BUILD #
#########
.PHONY: requirements develop build build-debug build-conda install

requirements:  ## install python dev and runtime dependencies
ifeq ($(OS),Windows_NT)
	Powershell.exe -executionpolicy bypass -noprofile .\ci\scripts\windows\make_requirements.ps1
else
	python -m pip install toml
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["build-system"]["requires"]))'`
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["develop"]))'`
endif

develop: requirements  ## install dependencies and build library
	python -m pip install -e .[develop]

build:  ## build the library
	python setup.py build build_ext --inplace

build-debug:  ## build the library ( DEBUG ) - May need a make clean when switching from regular build to build-debug and vice versa
	SKBUILD_CONFIGURE_OPTIONS="" DEBUG=1 python setup.py build build_ext --inplace

build-conda:  ## build the library in Conda
	python setup.py build build_ext --csp-no-vcpkg --inplace

build-conda-debug:  ## build the library ( DEBUG ) - in Conda
	SKBUILD_CONFIGURE_OPTIONS="" DEBUG=1 python setup.py build build_ext --csp-no-vcpkg --inplace

install:  ## install library
	python -m pip install .

#########
# LINTS #
#########
.PHONY: lint-py lint-cpp lint lints fix-py fix-cpp fix format check checks

lint-py:
	python -m isort --check csp/ examples/ setup.py
	python -m ruff check csp/ examples/ setup.py
	python -m ruff format --check csp/ examples/ setup.py

lint-cpp:
	# clang-format --dry-run -Werror -i -style=file `find ./cpp/ -name "*.*pp"`
	echo "C++ linting disabled for now"

lint-docs:
	python -m mdformat --check docs/wiki/ README.md examples/
	python -m codespell_lib docs/wiki/ README.md examples/ --skip "*.cpp,*.h"

# lint: lint-py lint-cpp  ## run lints
lint: lint-py lint-docs ## run lints

# Alias
lints: lint

fix-py:
	python -m isort csp/ examples/ setup.py
	python -m ruff format csp/ examples/ setup.py

fix-cpp:
	# clang-format -i -style=file `find ./cpp/ -name "*.*pp"`
	echo "C++ autoformatting disabled for now"

fix-docs:
	python -m mdformat docs/wiki/ README.md examples/
	python -m codespell_lib --write docs/wiki/ README.md examples/ --skip "*.cpp,*.h"

fix: fix-py fix-cpp fix-docs ## run autofixers

# alias
format: fix

check:
	check-manifest -v

# Alias
checks: check

#########
# TESTS #
#########
.PHONY: test-py test-cpp coverage-py test tests

TEST_ARGS :=
test-py: ## Clean and Make unit tests
	python -m pytest -v csp/tests --junitxml=junit.xml $(TEST_ARGS)

test-cpp: ## Make C++ unit tests
ifneq ($(OS),Windows_NT)
	for f in ./csp/tests/bin/*; do $$f; done || (echo "TEST FAILED" && exit 1)
else
	.\ci\scripts\windows\run_cpp_tests.bat
endif

coverage-py:
	python -m pytest -v csp/tests --junitxml=junit.xml --cov=csp --cov-report xml --cov-report html --cov-branch --cov-fail-under=80 --cov-report term-missing $(TEST_ARGS)

test: test-cpp test-py  ## run the tests

# Alias
tests: test

.PHONY: dockerup dockerps dockerdown initpodmanmac
ADAPTER := kafka
DOCKER := podman

initpodmanmac:
	podman machine stop
	podman machine set --cpus 4 --memory 8096
	podman machine start

dockerup:  ## spin up docker compose services for adapter testing
	$(DOCKER) compose -f ci/$(ADAPTER)/docker-compose.yml up -d

dockerps:  ## spin up docker compose services for adapter testing
	$(DOCKER) compose -f ci/$(ADAPTER)/docker-compose.yml ps

dockerdown:  ## spin up docker compose services for adapter testing
	$(DOCKER) compose -f ci/$(ADAPTER)/docker-compose.yml down

###########
# VERSION #
###########
.PHONY: show-version patch minor major

show-version:
	@ bump2version --dry-run --allow-dirty pyproject.toml --list | grep current | awk -F= '{print $$2}'

patch:
	bump2version patch

minor:
	bump2version minor

major:
	bump2version major

########
# DIST #
########
.PHONY: dist-py dist-py-sdist dist-py-wheel dist-py-cibw dist-check dist publish-py publish

dist-py: dist-py-sdist  # Build python dist
dist-py-sdist:
	rm -rf csp/lib/*
	python -m build --sdist -n

dist-py-wheel:
	python setup.py bdist_wheel $(EXTRA_ARGS)

dist-py-cibw:
	python -m cibuildwheel --output-dir dist $(EXTRA_ARGS)

dist-check:
	python -m twine check --strict dist/*

dist: clean build dist-py dist-check  ## Build dists

publish-py:  # Upload python assets
	python -m twine upload dist/* --skip-existing

publish: dist publish-py  ## Publish dists


.PHONY: notice

notice:
	printf 'CSP - Copyright 2024 Point72, L.P.\nThis project contains software with the below copyrights\n\n\n' > NOTICE
	for file in `find vcpkg_installed -name "copyright" | sort`; do echo $$file >> NOTICE && printf '\n\n' >> NOTICE && cat $$file >> NOTICE && printf '\n\n' >> NOTICE; done

#########
# CLEAN #
#########
.PHONY: deep-clean clean

deep-clean: ## clean everything from the repository
	git clean -fdx

clean: ## clean the repository
ifneq ($(OS),Windows_NT)
	rm -rf .coverage coverage cover htmlcov logs build dist wheelhouse *.egg-info
	rm -rf csp/lib csp/bin csp/include _skbuild
else
	del /s /q .coverage coverage cover htmlcov logs build dist wheelhouse *.egg-info
	del /s/ q csp\lib csp\bin csp\include _skbuild
endif

################
# Dependencies #
################
.PHONY: dependencies-mac dependencies-debian dependencies-fedora dependencies-vcpkg dependencies-win

dependencies-mac:  ## install dependencies for mac
	HOMEBREW_NO_AUTO_UPDATE=1 brew install bison cmake flex make ninja
	brew unlink bison flex && brew link --force bison flex

dependencies-debian:  ## install dependencies for linux
	apt-get install -y automake bison cmake curl flex ninja-build tar unzip zip

dependencies-fedora:  ## install dependencies for linux
	yum install -y automake bison ccache cmake curl flex perl-IPC-Cmd tar unzip zip

dependencies-vcpkg:  ## install dependencies via vcpkg
	cd vcpkg && ./bootstrap-vcpkg.sh && ./vcpkg install

dependencies-win:  ## install dependencies via windows
	choco install cmake curl winflexbison ninja unzip zip --no-progress -y

############################################################################################
# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
