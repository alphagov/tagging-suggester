import pytest

from src.utils.directories import (
    project_root_dir,
    data_dir,
    raw_data_dir,
    processed_data_dir
)

def test_project_root_dir():
    assert "tagging-suggester" in project_root_dir()

def test_data_dir():
    assert "tagging-suggester/data" in data_dir()

def test_raw_data_dir():
    assert "tagging-suggester/data/raw" in raw_data_dir()

def test_processed_data_dir():
    assert "tagging-suggester/data/processed" in processed_data_dir()

