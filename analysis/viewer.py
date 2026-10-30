
#Задача этого файла: найти логи, дать выбрать один из них, подготовить данные и показать таблицу.
# 1. Импортируем инструменты
import json
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Логи персонажа",
    layout="wide",
)


# 2. Настраиваем страницу
#st.set_page_config() задаёт настройки страницы:
# - page_title — название вкладки браузера.
# - layout="wide" — разрешение интерфейсу занимать широкую область страницы. Это полезно для нашей таблицы с большим количеством столбцов.
#А st.title() уже выводит заголовок внутри страницы.

# Это отличается от print(): print() пишет в терминал, а st.title() создаёт элемент интерфейса в браузере.

st.set_page_config(
    page_title="Логи персонажа",
    layout="wide",
)

st.title("Анализ диалогов")


# 3. Находим папку с логами
#__file__          Путь к выполняемому файлу viewer.py
#Path(__file__)    Объект для работы с этим путём
#.resolve()        Абсолютный путь с разрешением символических ссылок
#Первая .parent    Папку analysis, содержащую файл
#Вторая .parent    Папку проекта llm-paranoid


project_dir = Path(__file__).resolve().parent.parent
log_dir = project_dir / "logs"


# 4. Собираем список файлов и сортируем его
#log_dir.glob("*.jsonl")
#находит в папке logs пути, соответствующие шаблону:
#- * — произвольное имя;
#- .jsonl — нужное расширение.
#Полученные элементы — объекты Path.
#
#Для каждого пути:
#- path.stat() получает сведения о файле;
#- .st_mtime берёт время его последнего изменения.

log_files = sorted(
    log_dir.glob("*.jsonl"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

# 5. Проверяем, есть ли что показывать
#st.info() показывает информационное сообщение на странице.
#st.stop() останавливает текущее выполнение скрипта.
#
#Сам сервер Streamlit при этом продолжает работать.
#Мы просто не выполняем следующие строки, потому что выбирать файл из пустого списка невозможно.

if not log_files:
    st.info("В папке logs пока нет JSONL-файлов.")
    st.stop()


# 6. Даём пользователю выбрать лог
#st.selectbox() создаёт выпадающий список.
#
#Его параметры:
#- "Выбери запуск" — подпись элемента.
#- options=log_files — варианты выбора.
#- format_func — функция, определяющая, как показать каждый вариант.
#У нас варианты — объекты Path с полными путями. Но в интерфейсе удобнее видеть только имя файла:
#lambda path: path.name
#
#Например, для полного пути:
#/Users/igorgalkin/Projects/llm-paranoid/logs/run.jsonl
#path.name вернёт:
#run.jsonl

log_path = st.selectbox(
    "Выбери запуск",
    options=log_files,
    format_func=lambda path: path.name,
)


# 7. Читаем выбранный файл
#records — список, в который будем складывать прочитанные записи.
#
#log_path.open("r", encoding="utf-8")
#открывает файл:
#- "r" — для чтения;
#- encoding="utf-8" — с нужной кодировкой, в том числе для русского текста.
#with обеспечивает закрытие файла после выхода из блока, в том числе если внутри возникнет исключение.
#
#Цикл:
#for line in file:
#читает файл по строкам.
#Это подходит для JSONL, потому что каждая строка такого файла содержит отдельный JSON-объект.
#
#Проверка:
#if line.strip():
#пропускает пустые строки и строки из одних пробелов.

records = []

with log_path.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))


# 8. Превращаем записи в строки таблицы
rows = []

for turn_number, record in enumerate(records, start=1):
    before = record["state_before"]
    after = record["state_after"]

    rows.append(
        {
            "Ход": turn_number,
            "Сообщение пользователя": record["user_message"],
            "Ответ модели": record["reply"],
            "Тревога до": before["anxiety_level"],
            "Тревога после": after["anxiety_level"],
            "Изменение тревоги": (
                after["anxiety_level"] - before["anxiety_level"]
            ),
            "Confidence до": before["confidence"],
            "Confidence после": after["confidence"],
            "Интерпретация до": before["interpretation"],
            "Интерпретация после": after["interpretation"],
            "Намерение ответа": record["response_intent"],
        }
    )

# 9. Показываем количество ходов
#st.caption() выводит текст в интерфейсе небольшим шрифтом.

st.caption(f"Записано ходов: {len(rows)}")

# 10. Показываем таблицу
#st.dataframe() принимает подготовленные данные и создаёт интерактивную таблицу.
#Название функции не означает, что ей обязательно нужен объект pandas: список словарей тоже подходит.


st.dataframe(
    rows,
    hide_index=True,
    width="stretch",
    row_height=140,
)
