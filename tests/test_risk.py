from src.risk import calculate_risk


def test_breach_can_only_raise_risk() -> None:
    clean = calculate_risk(4, {"status": "checked", "found": False, "count": 0})
    breached = calculate_risk(4, {"status": "checked", "found": True, "count": 1000})
    assert clean["score"] == 0
    assert breached["score"] >= 60


def test_unavailable_hibp_marks_result_as_incomplete() -> None:
    result = calculate_risk(2, {"status": "unavailable", "found": None, "count": None})
    assert result["complete"] is False
    assert result["breach_component"] is None

