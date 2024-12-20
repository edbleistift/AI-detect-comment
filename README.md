## Процесс обработки и обучения модели

1. **Загрузка и предварительная обработка данных**  
   На первом этапе происходит загрузка данных из файла `cleaned_kaspi_reviews.csv`. После этого удаляются ненужные столбцы, и выполняется фильтрация данных: 
   - Оставляются только русскоязычные отзывы.
   - Создаются метки для классификации, где `1` — это хороший отзыв, а `0` — плохой.

2. **Токенизация текста**  
   Для подготовки текстов к подаче в модель используется токенизатор BERT (`BertTokenizer`). Он преобразует текст в формат, который модель BERT может эффективно обработать, добавляя необходимые токены и выполняя паддинг.

3. **Создание и обучение модели**  
   На этом этапе загружается модель BERT для задачи классификации текстов из библиотеки `transformers`. Модель обучается на подготовленных данных, и на протяжении нескольких эпох происходит оптимизация её параметров.

4. **Оценка модели**  
   После завершения тренировки модель оценивается на тестовой выборке, где вычисляется точность предсказаний. Это позволяет нам узнать, насколько хорошо модель справляется с задачей классификации отзывов.

5. **Сохранение модели**  
   После тренировки обученная модель и токенизатор сохраняются в директорию `saved_model`, что позволяет в дальнейшем использовать их для предсказаний без необходимости повторного обучения.

## Файлы проекта

- model.py — Основной скрипт для обучения модели.
- test_model.py — Скрипт для тестирования модели.
- saved_model/ — Папка с сохранённой моделью и токенизатором.

## Данные
**Модель обучена на основе ~120к данных с комментариями и оценками за товары**
