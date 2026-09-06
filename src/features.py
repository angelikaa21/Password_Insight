from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable

import pandas as pd


FEATURE_LABELS = {
    "length": "Długość hasła",
    "unique_count": "Liczba unikalnych znaków",
    "unique_ratio": "Udział unikalnych znaków",
    "lowercase_count": "Liczba małych liter",
    "uppercase_count": "Liczba wielkich liter",
    "digit_count": "Liczba cyfr",
    "special_count": "Liczba znaków specjalnych",
    "category_count": "Liczba kategorii znaków",
    "lowercase_ratio": "Udział małych liter",
    "uppercase_ratio": "Udział wielkich liter",
    "digit_ratio": "Udział cyfr",
    "special_ratio": "Udział znaków specjalnych",
    "repeated_char_count": "Liczba powtórzonych znaków",
    "max_char_run": "Najdłuższa seria jednego znaku",
    "sequence_length": "Najdłuższa prosta sekwencja",
    "keyboard_sequence_length": "Najdłuższa sekwencja klawiaturowa",
    "repeated_fragment_count": "Powtarzające się fragmenty",
    "common_pattern_count": "Popularne wzorce",
    "charset_entropy": "Entropia wynikająca z alfabetu",
}

FEATURE_NAMES = list(FEATURE_LABELS)

COMMON_PATTERNS = (
    "password",
    "pass",
    "admin",
    "qwerty",
    "asdf",
    "welcome",
    "letmein",
    "login",
    "haslo",
    "abc",
    "123",
)

KEYBOARD_ROWS = (
    "1234567890",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
)


def _safe_ratio(value: int, length: int) -> float:
    return value / length if length else 0.0


def _max_char_run(password: str) -> int:
    if not password:
        return 0
    longest = current = 1
    for previous, current_char in zip(password, password[1:]):
        current = current + 1 if current_char == previous else 1
        longest = max(longest, current)
    return longest


def _max_simple_sequence(password: str) -> int:
    if not password:
        return 0
    lowered = password.lower()
    longest = current = 1
    previous_step: int | None = None
    for previous, current_char in zip(lowered, lowered[1:]):
        if previous.isalnum() and current_char.isalnum():
            step = ord(current_char) - ord(previous)
            if step in (-1, 1):
                current = current + 1 if previous_step == step else 2
                longest = max(longest, current)
                previous_step = step
                continue
        current = 1
        previous_step = None
    return longest


def _max_keyboard_sequence(password: str) -> int:
    lowered = password.lower()
    longest = 0
    for row in KEYBOARD_ROWS:
        for candidate in (row, row[::-1]):
            for start in range(len(candidate)):
                for end in range(start + 3, len(candidate) + 1):
                    fragment = candidate[start:end]
                    if fragment in lowered:
                        longest = max(longest, len(fragment))
    return longest


def _repeated_fragment_count(password: str) -> int:
    lowered = password.lower()
    repeated: set[str] = set()
    for size in (2, 3, 4):
        fragments = Counter(
            lowered[index : index + size]
            for index in range(max(0, len(lowered) - size + 1))
        )
        repeated.update(fragment for fragment, count in fragments.items() if count > 1)
    return len(repeated)


def _common_pattern_count(password: str) -> int:
    lowered = password.lower()
    count = sum(pattern in lowered for pattern in COMMON_PATTERNS)
    count += int(bool(re.search(r"(?:19|20)\d{2}", password)))
    return count


def _alphabet_size(password: str) -> int:
    size = 0
    if any(char.islower() for char in password):
        size += 26
    if any(char.isupper() for char in password):
        size += 26
    if any(char.isdigit() for char in password):
        size += 10
    if any(not char.isalnum() for char in password):
        size += len(string.punctuation)
    return max(size, 1)


def extract_features(password: str) -> dict[str, float]:
    """Zwraca wyłącznie cechy numeryczne; wynik nie zawiera treści hasła."""
    if not isinstance(password, str):
        raise TypeError("Hasło musi być ciągiem znaków.")

    length = len(password)
    lowercase = sum(char.islower() for char in password)
    uppercase = sum(char.isupper() for char in password)
    digits = sum(char.isdigit() for char in password)
    special = sum(not char.isalnum() for char in password)
    unique = len(set(password))
    categories = sum(value > 0 for value in (lowercase, uppercase, digits, special))
    repeated = sum(count - 1 for count in Counter(password).values() if count > 1)

    features: dict[str, float] = {
        "length": float(length),
        "unique_count": float(unique),
        "unique_ratio": _safe_ratio(unique, length),
        "lowercase_count": float(lowercase),
        "uppercase_count": float(uppercase),
        "digit_count": float(digits),
        "special_count": float(special),
        "category_count": float(categories),
        "lowercase_ratio": _safe_ratio(lowercase, length),
        "uppercase_ratio": _safe_ratio(uppercase, length),
        "digit_ratio": _safe_ratio(digits, length),
        "special_ratio": _safe_ratio(special, length),
        "repeated_char_count": float(repeated),
        "max_char_run": float(_max_char_run(password)),
        "sequence_length": float(_max_simple_sequence(password)),
        "keyboard_sequence_length": float(_max_keyboard_sequence(password)),
        "repeated_fragment_count": float(_repeated_fragment_count(password)),
        "common_pattern_count": float(_common_pattern_count(password)),
        "charset_entropy": float(length * math.log2(_alphabet_size(password))) if length else 0.0,
    }
    return features


def features_frame(passwords: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame((extract_features(password) for password in passwords), columns=FEATURE_NAMES)

