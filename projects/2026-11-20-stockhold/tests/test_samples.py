import json

from conftest import ROOT
from stockhold.demo import run


def test_scenario_matches_expected_output():
    scenario = json.loads((ROOT / "samples" / "scenario.json").read_text())
    expected = (ROOT / "samples" / "scenario_expected.txt").read_text().splitlines()
    assert run(scenario) == expected


def test_catalog_is_valid():
    items = json.loads((ROOT / "samples" / "catalog.json").read_text())
    assert {"SNKR-01", "VNYL-07", "KBD-65"} == {i["sku"] for i in items}
    assert all(i["total"] >= 1 and i["per_customer_limit"] >= 1 for i in items)
