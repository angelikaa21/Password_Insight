# Password Insight

Password Insight jest lokalnym systemem badawczym służącym do oceny siły i ryzyka
haseł. System łączy klasyfikację na podstawie 19 cech strukturalnych, kontrolę obecności
hasła w HIBP Pwned Passwords, jawną agregację ryzyka oraz lokalne wyjaśnienia SHAP.

## Funkcje systemu

- klasyfikacja siły hasła w pięciostopniowej skali,
- analiza cech strukturalnych,
- sprawdzenie wystąpienia hasła w znanych naruszeniach,
- wyznaczenie końcowego wskaźnika ryzyka,
- prezentacja rozkładu prawdopodobieństwa klas,
- lokalne wyjaśnienie wyniku modelu,
- rekomendacje wynikające z wykrytych cech i zagrożeń.

## Uruchomienie w Windows

1. Zainstaluj 64-bitowy Python 3.11+ i zaznacz `Add Python to PATH`.
2. Rozpakuj cały katalog projektu.
3. Uruchom `start_windows.bat`.
4. Przy pierwszym uruchomieniu zaczekaj na instalację wymaganych bibliotek.
5. Otwórz `http://localhost:8501`, jeśli przeglądarka nie uruchomi się automatycznie.
6. Aby zatrzymać aplikację, wróć do okna terminala i naciśnij `Ctrl+C`.

## Uruchomienie ręczne

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Model i eksperyment

Model Random Forest został wytrenowany na zbalansowanej próbce PWLDS po usunięciu
duplikatów oraz rekordów o sprzecznych etykietach. Eksperyment obejmuje podział
70:15:15, trzy ziarna losowe oraz porównanie Logistic Regression, Random Forest,
XGBoost i zxcvbn. Wyniki są zapisywane w katalogu `results`.

Pełne odtworzenie eksperymentu w Windows uruchamia plik:

```text
prepare_research_model.bat
```

Skrypt pobiera dane, przygotowuje próbkę, przeprowadza trening i ewaluację, a następnie
tworzy `WYNIKI_BADAWCZE.zip`. Archiwum nie zawiera surowych haseł.

Ręczne wykonanie kolejnych etapów:

```powershell
python -m pip install -r requirements-research.txt
python scripts/prepare_pwlds.py --sample-size 100000
python scripts/train_models.py data/processed/pwlds_sample.csv
python scripts/package_research_results.py
```

## Testy

```powershell
python -m pip install pytest
python -m pytest
```

## Prywatność

- Hasło nie jest zapisywane w bazie danych ani w plikach wynikowych.
- Formularz jest czyszczony po wykonaniu analizy.
- Do HIBP wysyłany jest tylko pięcioznakowy prefiks skrótu SHA-1.
- Porównanie pełnego skrótu odbywa się lokalnie.
- Niedostępność HIBP jest oznaczana jako niepełna ocena ryzyka.

## Agregacja ryzyka

Ryzyko strukturalne jest obliczane jako `100 - 25 × klasa_siły`. Dla hasła wykrytego
w HIBP składowa naruszenia wynosi `min(100, 60 + 10 × log10(liczba_wystąpień))`.
Wynik końcowy stanowi większa z obu wartości, dlatego potwierdzone naruszenie może
wyłącznie zwiększyć poziom ryzyka.
