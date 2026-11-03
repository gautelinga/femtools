"""A suite that spawns mpirun is also slow."""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("mpi") and not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)
