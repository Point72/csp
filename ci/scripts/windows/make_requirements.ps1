python -m pip install toml
python -m pip install $(python -c "import toml; c = toml.load('pyproject.toml'); print('\n'.join(c['build-system']['requires']))")
python -m pip install $(python -c "import toml; c = toml.load('pyproject.toml'); print('\n'.join(c['project']['optional-dependencies']['develop']))")