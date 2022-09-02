#!/bin/bash

pdoc --html istat -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/istat/* docs
