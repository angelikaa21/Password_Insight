from src.features import FEATURE_NAMES, extract_features


def test_feature_vector_has_expected_shape() -> None:
    features = extract_features("Przyklad!2026")
    assert list(features) == FEATURE_NAMES
    assert len(features) == 19
    assert "password" not in features


def test_repetitions_and_sequences_are_detected() -> None:
    features = extract_features("aaaa1234")
    assert features["max_char_run"] == 4
    assert features["sequence_length"] >= 4
    assert features["common_pattern_count"] >= 1


def test_empty_password_is_supported_by_extractor() -> None:
    features = extract_features("")
    assert features["length"] == 0
    assert features["unique_ratio"] == 0

