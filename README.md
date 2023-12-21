# mlops_hw

В качестве игрушечного проекта используется логистическая регрессия на Iris
датасете.

## Предварительно

1. Скопировать репозиторий
2. Создать виртуальное окружение
3. Находясь в репозитории и активировав виртуальное окружение, установить
   зависимости

```
poetry install
```

## Обучение

Для логирования необходимо поднять сервер `MLFlow`.

```
mlflow server --host 127.0.0.1 --port 8080
```

Либо другой адрес и порт, для чего необходимо поменять `train.mlflow_server` в
`iris_classifier/conf/config.yaml`.

Далее запустить скрипт обучения

```
python iris_classifier/train.py
```

По окончании выполнения появятся файлы модели в форматах `.sav` и `.onnx`, а
также результат от `MLFlow` в папке `model`. Также появятся графики результатов
валидации, они также все буду залогированы `MLFlow`.

## Инференс (локально)

Для локального инференса (как в первом домашнем задании):

```
python iris_classifier/infer.py
```

В результате будут выведен отчёт с метриками на тесте, а в репозитории появится
`predictions.csv` с предсказаниями на тестовой выборке.

## Инференс с MLFlow Models

Сначала необходимо поднять сервер для инференса.

```
mlflow models serve -m model --env-manager local --host 127.0.0.1
```

Если нужно поменять адрес или порт, перед этим так же нужно поменять конфиг
`infer.mlflow_server` в `iris_classifier/conf/config.yaml`. Далее запустить
скрипт

```
python iris_classifier/run_server.py
```

Скрипт отправит тестовый запрос на сервер, в ответ будет получен результат
работы в виде

```
{'output_label': 'virginica',
'output_probability': {
  'setosa': 0.0006740661920048296,
  'versicolor': 0.08054611086845398,
  'virginica': 0.9187798500061035
  }
}
```

## Triton
### Как запускать
Для запуска Triton надо сначала перейти в соответствующую папку
```
cd triton
```

Стартануть Docker
```
sudo systemctl start docker
```

И поднять непосредственно сам контейнер

```
docker-compose up
```

Для запуска клиента сервера и прогонки тестов запускаем скрипт (из той же папки!)

```
python cleint.py
```

Если тесты пройдут гладко, будет выведен результат выполненого запроса.

### Отчёт
- Системная конфигурация
  + OS и версия: Fedora 36
  + Модель CPU: AMD Ryzen 5 5600U with Radeon Graphics
  + Количество vCPU и RAM при котором собирались метрики: 3 vCPU, 16 GB RAM
- Описание решаемой задачи: обучение логистической регрессии на всем известном датасете про ирисы Iris Dataset.
- Описание структуры model_repository: .
- Метрики
  + До оптимизации
    * throughput:
    * latency:
  + После оптимизации
    * throughput:
    * latency:
- Объяснение мотивации и выбора: как на семинаре, бинарным поиском сходился к оптимальной задержке в динамическом батчинге. Метрики показали, что 1 мс - оптимально (самый большой throughput и наименьший latency). 
