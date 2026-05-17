import pytest

from sionna_measurement_sim.phy.modules import PHY_REGISTRY, get_phy_module


def test_builtin_phy_modules_are_registered():
    assert sorted(PHY_REGISTRY) == ["custom_ofdm", "nr_pusch", "nr_srs"]
    assert get_phy_module("custom_ofdm").standard == "custom_ofdm"
    assert get_phy_module("nr_pusch").standard == "nr_pusch"
    assert get_phy_module("nr_srs").standard == "nr_srs"


def test_unknown_phy_module_has_clear_error():
    with pytest.raises(ValueError, match="Unsupported PHY standard"):
        get_phy_module("not_a_phy")
