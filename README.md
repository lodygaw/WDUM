# Wprowadzenie do Uczenia Maszyn

Rozwiązanie zadania projektowego, semestr 2022Z

## Opis zadania

Celem projektu było zaimplementowanie procedury klasyfikacji opartej na algorytmie [ROCKET](https://arxiv.org/abs/1910.13051) w wersji dla szeregów czasowych wielu zmiennych, przy założeniu, że szeregi mogą mieć różne długości. Do testowania algorytmu należało posłużyć się 10 dowolnymi zbiorami danych ze strony https://timeseriesclassification.com (tylko szeregi wielu zmiennych). Należało zapewnić przycięcie szeregów czasowych metodą stratyfikacji po klasach. W każdej klasie szeregi czasowe miały zostać przycięte do następujących długosci:
- `1/3` instancji o dowolnej długości z przedziału `<10%, 40%>`
- `1/3` instancji o dowolnej długości z przedziału `(40%,70%>`
- `1/3` instancji o dowolnej długości z przedziału  `(70%, 100%>`

Należało zaproponować metodę uzupełnienia danych w przyciętych szeregach, która pozwoli uzyskać wyższą dokładność klasyfikacji niż padding.

## Implementacja

Rozwiązanie zostało zaimplementowane w języku programowania [Julia](https://julialang.org). Instrukcja instalacji oraz linki do pobrania Julii znajdują się [tutaj](https://julialang.org/downloads/).

Rozwiązanie projektu można podzielić na kilka części:
- `rocket.jl` - autorska implementacja algorytmu ROCKET
- `forecast.jl` - implementacje metod uzupełniania przyciętych szeregów czasowych
- `datasets.jl` - implementacje funkcji obsługujacych zbiory danych, w tym pobieranie, wczytywanie i przycinanie.

Wszystkie elementu projektu są udostępnione w ramach modułu WDUM. Aby móc użyć modułu należy wcześniej aktywować `Project.toml`. 
Najłatwiej zrobić to w następujący sposób:
```julia
using Pkg
Pkg.activate(".")               # path to Project.toml

using WDUM.ROCKET
using WDUM.Forecasts
using WDUM.Datasets
```
## Użycie 
### WDUM.Datasets
#### Pobieranie zbiorów danych
```julia
download_dataset("BasicMotions", "../custom_directory")       # using custom directory for data
download_dataset("BasicMotions")                              # defaults data directory to "../data"
```
#### Wczytywanie zbiorów danych
```julia
X_train, y_train = load_dataset("BasicMotions", "train")      # loading only train data
X_test, y_test = load_dataset("BasicMotions", "test")         # loading only test data

X_train, y_train, X_test, y_test = load_dataset("BasicMotions")
```
#### Przycinanie zbiorów danych
Do przycinania zbiorów danych stworzono prostą strukturę `Interval` do zdefiniowania przedziałów przycięcia.
```julia
interval = Interval{Closed, Closed}(0.1, 0.4)
```

Funkcja `trim_timeseries` przyjmuje w argumencie wektor struktur `Interval`. Funkcja dokonuje przycięcia w równym stosunku w podanych przedziałach ze stratyfikacją po klasach. Obcięte elementy są zastępowane przez wartość `NaN`.
```julia
ranges = [Interval{Closed, Closed}(0.1, 0.4),
          Interval{Open, Closed}(0.4, 0.7),
          Interval{Open, Closed}(0.7,1.0)]

X_trimmed = trim_timeseries(X, y, ranges)
```
#### Funkcje pomocnicze
```julia
# dummy function for printing how many instances are in intervals from task description, in all classes
show_stratification(X_trimmed, y)

# dummy function for plotting whole SINGLE timeseries (expects ts to be matrix in dimension n_features x n_timesteps
plot_timeseries(ts)
```

### WDUM.ROCKET

```julia
using WDUM.ROCKET

rocket = Rocket(num_kernels=10_000, normalize=true, precision=Float32)
```

Algorytm może być uruchomiony dla dowolnej precyzji liczb zmiennoprzecinkowych, które można skonfigurować przy pomocy argumentu `precision`.

```julia
# fit!(r::Rocket{FLOAT, NKERNELS, NORMALIZE}, X::Array{FLOAT, 3}, seed=nothing)
fit!(rocket, X)

# transform!(r::Rocket{FLOAT, NKERNELS, NORMALIZE}, X::Array{FLOAT, 3})
X_ = transform!(rocket, X)
```
Aktualna implementacja zakłada format danych wejściowych `X` jako macierz trójwymiarową o wymiarach `n_timeseries × n_features × n_timesteps`.

### WDUM.Forecasts
#### Mean - padding wartością średnią
```julia
target_length = 100                               # final length of timeseries

# forecast(type::Type{Mean}, X::AbstractArray{T,3}, target_length::Int)
X_ = forecast(Mean, X, target_length)
```
#### NaiveSingle - autorska metoda prognozowania na podstawie danych ze prognozowanego szeregu czasowego
```julia
target_length = 100                               # final length of timeseries
window = 4                                        # lengths of timeseries sections for matching and forecasting
overlap = 3                                       # length of timeseries section which will be matched with original timeseries 

# forecast(type::Type{NaiveSingle}, X::AbstractArray{T,3}, target_length::Int, window::Int, overlap::Int)
X_ = forecast(NaiveSingle, X, target_length, window, overlap)
```
Liczba prognozowanych kroków czasowych to `window` - `overlap`.
#### NaiveMultiple - autorska metoda prognozowania na podstawie danych z całego zbioru
```julia
target_length = 100                               # final length of timeseries
max_overlap = 40                                  # maximal allowed overlap
min_overlap = 5                                   # minimal allowed overlap
threshold = 0.4                                   # matching error threshold for using certain timeseries for forecasting

# forecast(type::Type{NaiveMultiple}, X::AbstractArray{T,3}, target_length::Int, max_overlap::Int, min_overlap::Int, threshold::T)
X_ = forecast(NaiveMultiple, X, target_length, max_overlap, min_overlap, threshold)
```
### Objaśnienie działania autorskich metod prognozowania
Obie metody zostały opracowane w zamyśle do danych wśród których nie da się zaobserwować trendu ani cykliczności - czyli nienadającymi się do prognozowania metodami statystycznymi. Metoda `NaiveMultiple` jest tak naprawdę rozwinięciem metody `NaiveSingle`, która w rzeczywistości bardzo rzadko daje wyniki lepsze od paddingu. Prawdopodobnie dostoswanie części algorytmu z metody `NaiveMultiple` do kontekstu metody `NaiveSingle` pomogłoby zwiększyć jej użyteczność.

Zarówno metoda `NaiveSingle`, jak `NaiveMultiple` bazują na porównywaniu odcinka szeregu czasowego z progozowanym szeregiem czasowym. Aktualnie wykorzystywaną miarą porównania jest *Mean Average Error*.
#### NaiveSingle
Metoda polega na podzieleniu prognozowanego szeregu czasowego na odcinki o długości równej `window`. 
Sprawdzane są wszystkie możliwe kombinacje tj. szereg czasowy `ts` o długości `10` i `window` równym `5` będzie podzielony na odcinki (indeksowanie od 1.):
- `ts[1:5]`
- `ts[2:6]`
- `ts[3:7]`
- `ts[4:8]`
- `ts[5:9]`
- `ts[6:10]`

Po podziale wszystkie odcinki będą "porównane" z końcówką prognozowanego szeregu czasowego. Liczba porównanych kroków czasowych zależy od argumentu `overlap`.
```julia
section = ts[1:5]
overlap = 3
error = loss(ts[(end - overlap):end], section[1:overlap]) 
```
Odcinek, którego błąd porównania będzie **najmniejszy** zostanie wykorzystany do prognozowania:
```julia
best_section = ts[1:5]
window = 5
overlap = 3
foecast_length = window - overlap
push!(ts, best_section[(end - forecast_length):end])
```
#### NaiveMultiple
Metoda polega na porównywaniu odcinków wszystkich szeregów czasowych ze zbioru z prognozowanym szeregiem czasowym.
