# Использование PPO в задаче управления самолетом в продольном канале 

![Py_Ver](https://img.shields.io/badge/Python-3.9-brightgreen?style=plastic&color=blue)

## Описание

В данном репозитории находится реализация работы:
[Todo: paper link].

`F16model` --- находится нелинейная модель F16 для продольного канала.

`control` --- реализация RL алгоритмов

`logs` --- туда сохраняются логи

`runs` --- логгирование при обучении алгоритмов 

`runs\models` --- веса моделей которые можно поднять

`apps\trim-app.py` --- простенькая гуишка для нахождения балансировочного состояния 

## Установка 

```bash
pip install -r requirements.txt
python setup.py develop
```

