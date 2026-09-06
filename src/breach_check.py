from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Callable

import requests


HIBP_RANGE_URL = "https://api.pwnedpasswords.com/range/{prefix}"


@dataclass(frozen=True)
class BreachResult:
    status: str
    found: bool | None
    count: int | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_pwned_password(
    password: str,
    timeout: float = 5.0,
    request_get: Callable[..., object] = requests.get,
) -> BreachResult:
    """Sprawdza HIBP bez przesyłania hasła ani pełnego skrótu."""
    digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]

    try:
        response = request_get(
            HIBP_RANGE_URL.format(prefix=prefix),
            headers={"Add-Padding": "true", "User-Agent": "PasswordRiskResearch/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException:
        return BreachResult(
            status="unavailable",
            found=None,
            count=None,
            message="Nie udało się połączyć z usługą HIBP.",
        )

    for line in response.text.splitlines():
        candidate, separator, count = line.partition(":")
        if separator and candidate.strip().upper() == suffix:
            occurrences = int(count.strip())
            return BreachResult(
                status="checked",
                found=True,
                count=occurrences,
                message=f"Hasło wystąpiło w znanych naruszeniach {occurrences:,} razy.".replace(",", " "),
            )

    return BreachResult(
        status="checked",
        found=False,
        count=0,
        message="Nie znaleziono hasła w aktualnej odpowiedzi HIBP.",
    )

