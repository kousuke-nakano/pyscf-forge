#!/usr/bin/env bash
  
set -e

cd ./pyscf
pytest -k 'not _slow' -s -v tools/test/test_trexio.py
