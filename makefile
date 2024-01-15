.PHONY: remove install

all: install
remove:
	@echo "Removing files..."
	# Add commands to remove files here
	python3 -m pip uninstall tutel -y

install: remove
	@echo "Installing..."
	# Add commands to install here
	python3 ./setup.py install --user
