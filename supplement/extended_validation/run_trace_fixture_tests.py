"""Run unchanged trace fixtures with published global settings only."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'trace'))
import engine

parameters = json.loads((HERE / 'trace_parameters_20261109.json').read_text())
original = engine.original_parameters
try:
    engine.original_parameters = lambda: ({**parameters['pool'], 'mqhoa': parameters['mqhoa']}, {'PSO': parameters['PSO']})
    spec = importlib.util.spec_from_file_location('published_trace_fixtures', HERE / 'trace/tests.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    passed = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
finally:
    engine.original_parameters = original
if not passed:
    raise SystemExit(1)
