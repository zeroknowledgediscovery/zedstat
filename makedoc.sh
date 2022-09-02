#!/bin/bash

pdoc --html zedstat -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/zedstat/* docs
