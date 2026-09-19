import pytest
from pydantic import ValidationError

from visit_ranker.config import load_config


def test_split_boundaries_are_in_order(cfg):
    assert cfg.start_date < cfg.train_end < cfg.validation_end <= cfg.end_date


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown profile"):
        load_config(profile="nope")


def test_config_is_frozen(cfg):
    with pytest.raises(ValidationError):
        cfg.seed = 1


def test_full_profile_has_enough_cities_defined():
    cfg = load_config(profile="full")
    assert len(cfg.cities) == cfg.profile.n_cities
