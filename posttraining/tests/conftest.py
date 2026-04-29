# coding=utf-8
"""Pytest configuration and shared fixtures for wedlm_train tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Return the test device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def set_random_seed():
    """Set deterministic random seeds before each test."""
    torch.manual_seed(42)
