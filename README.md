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


## Формулы, использованные в модели

1. **Формула для создания меток классов**  
   Для классификации отзывов на хорошие и плохие используется простая бинарная метка на основе рейтинга отзыва:

   - Если рейтинг отзыва $x$ меньше или равен 2, то метка $0$ (плохой отзыв).
   - Если рейтинг отзыва $x$ больше 2, то метка $1$ (хороший отзыв).

   Формула:
   $$
   \text{label} = 
   \begin{cases}
   0, & \text{если } x \leq 2 \\
   1, & \text{если } x > 2
   \end{cases}
   $$

2. **Формула для потерь в процессе обучения**  
   В процессе обучения модели используется функция потерь (loss function), которая измеряет ошибку предсказания модели. Для задачи классификации используется кросс-энтропийная потеря (cross-entropy loss), которая рассчитывается как:

   $$
   L = -\sum_{i=1}^{N} \left( y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right)
   $$

   Где:
   - $L$ — кросс-энтропийная потеря,
   - $N$ — количество примеров в выборке,
   - $y_i$ — истинная метка для $i$-го примера,
   - $p_i$ — вероятность, предсказанная моделью для $i$-го примера.

3. **Оптимизация с использованием AdamW**  
   Для оптимизации модели используется алгоритм AdamW (вариант Adam с weight decay). Обновление весов производится по следующей формуле:

   $$
   \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t
   $$

   Где:
   - $\theta_t$ — параметры модели на шаге $t$,
   - $\eta$ — скорость обучения,
   - $m_t$ — момент первого порядка (первый момент),
   - $v_t$ — момент второго порядка (второй момент),
   - $\epsilon$ — малое число для предотвращения деления на ноль.


