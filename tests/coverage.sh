#!/bin/bash
pytest --cov=tbmodels --cov-config ../.coveragerc --cov-report=html "$@"
