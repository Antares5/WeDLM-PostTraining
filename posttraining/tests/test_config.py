# coding=utf-8
"""Unit tests for config loading and validation."""

import tempfile
import os
from wedlm_train.config import SFTConfig, DPOConfig, GSPOConfig, from_yaml


def _write_yaml(path, data):
    import yaml
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestConfigLoading:
    def test_from_yaml_sft(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("training_mode: sft\nmodel_path: test-model\n")
            f.flush()
            config = from_yaml(f.name)
            assert isinstance(config, SFTConfig)
            assert config.model_path == "test-model"
        os.unlink(f.name)

    def test_from_yaml_dpo(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("training_mode: dpo\ndpo_beta: 0.5\n")
            f.flush()
            config = from_yaml(f.name)
            assert isinstance(config, DPOConfig)
            assert config.dpo_beta == 0.5
        os.unlink(f.name)

    def test_from_yaml_gspo(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("training_mode: gspo\ngspo_group_size: 8\n")
            f.flush()
            config = from_yaml(f.name)
            assert isinstance(config, GSPOConfig)
            assert config.gspo_group_size == 8
        os.unlink(f.name)


class TestConfigValidation:
    def test_invalid_mode(self):
        try:
            SFTConfig(training_mode="invalid")
            assert False
        except ValueError:
            pass

    def test_invalid_dpo_block_reduce(self):
        try:
            DPOConfig(training_mode="dpo", dpo_block_reduce="invalid")
            assert False
        except ValueError:
            pass

    def test_gspo_group_size_min(self):
        try:
            GSPOConfig(training_mode="gspo", gspo_group_size=1)
            assert False
        except ValueError:
            pass
