# Notes from the meeting

## Setup

* ~~**true train/val/test**~~
* ~~**wracamy do zbiorów meta dla enkodera i zbiorów do testowania metody**~~
* **~~fitujemy hiperparametry, 3 punkty dla softmax~~, ~~i pełny fit dla stepu (przemyśleć analityczny wzór na stepsize), tutaj ważne, że robić to w środku, żeby oszczędzić czas~~**
* ~~**pretrainy liltaba per hiperparametry i target weighting**~~

* dobra weryfikacja lossu enkodera per zadanie i preselekcja danych do treningu

## Eksperymenty

* ablation study
    * ~~analiza wielkość datasetu do wyniku~~
    * ~~rozkład wag w modelach a score adaptivee (entropia)~~
    * ~~L1 różnic między static a dynamic~~
        * podział modeli na grupy względem wag i wtedy analiza różnic (na potem, bo to niewiele da)
    * ~~raportowanie step size. alpha~~
* **nowa metryka - 0 to wynik modelu statycznego, 1 to wynik modelu idealnego i w ten sposób skalujemy score (bazowana na accuracy)**
* AutoGluon, zapisać modele i wagi

## Co zrobić

Raportujemy

* modele i wagi z autgluona
* statystyki z datasetu
* nową metrykę
* step size optymalny, na val i na test

Pokazujemy

* nową metrykę
* L1 różnic
* pokazujemy entropię modeli
* pokazujemy jakie modele i ile i jaka średnia waga
* pokazujemy średni step size i różnicę pomiędzy val a test