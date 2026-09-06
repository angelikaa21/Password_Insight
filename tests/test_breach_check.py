import hashlib

from src.breach_check import check_pwned_password


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_only_prefix_is_used_in_request() -> None:
    password = "TajneHaslo!2026"
    digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    calls: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls["url"] = url
        calls["kwargs"] = kwargs
        return FakeResponse(f"{digest[5:]}:17\nABCDEF:2")

    result = check_pwned_password(password, request_get=fake_get)
    assert result.found is True
    assert result.count == 17
    assert digest[:5] in str(calls["url"])
    assert digest not in str(calls["url"])
    assert password not in str(calls)

