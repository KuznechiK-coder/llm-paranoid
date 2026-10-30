import json
import argparse
from datetime import datetime
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from character import CHARACTER_PROFILE


def build_system_prompt(
    profile,
    state,
    response_intent,
):
    system_prompt = (
        "Ты ведёшь диалог от лица описанного персонажа. \n"
        "Отвечай его репликами, а не рассказом о его поведении.\n\n"
        + profile
        + "Ниже словарь описывает состояние персонажа на текущий момент\n"
        + "anxiety_level - уровень тревоги от 0 до 100, где 0 это спокойное состояние, 100 это максимально тревожное состояние\n"
        + "interpretation - субъективный взгляд персонажа на происходящий диалог, этот взгляд может быть ошибочным\n"
        + "confidence - уверенность персонажа в своей интерпретаци происходящей в диалоге ситуации, "
        + "где 0 - абсолютно не уверен, 100 - полностью уверен."
        + f"{json.dumps(state, ensure_ascii=False, indent=2,)}\n"
        + f"\n\nНамерение ответа: {response_intent}\n\n"
        + "Реплика персонажа должна соответствовать его намерению\n"
        + "Не выводи служебные данные.\n\n"
    )
    return system_prompt


def build_appraisal_prompt(profile, state):
    return f"""
Ты определяешь внутреннее состояние персонажа после последнего
сообщения пользователя и цель его следующей реплики.

Это служебная оценка. Профиль описывает персонажа, которого ты оцениваешь.
Указания профиля о манере речи относятся к его репликам.
В этом задании верни только JSON, без самой реплики персонажа.

ПРОФИЛЬ ПЕРСОНАЖА

{profile}

СОСТОЯНИЕ ДО ПОСЛЕДНЕГО СООБЩЕНИЯ ПОЛЬЗОВАТЕЛЯ

{json.dumps(state, ensure_ascii=False, indent=2)}

ОЦЕНКА СИТУАЦИИ

Учитывай предыдущее состояние, всю доступную историю и последнее
сообщение пользователя.

Различай нейтральное сообщение, неоднозначный сигнал, собственную
ошибку, сравнение с другой моделью и прямую угрозу.
В ситуации могут одновременно присутствовать несколько мотивов.

Утверждение пользователя об ошибке не доказывает её существование.
Отличай подтверждённую ошибку от спорной оценки и необоснованного
обвинения, опираясь на доступное содержание разговора.

Отделяй события диалога от предположений персонажа.
Персонаж может ошибаться в намерениях пользователя, преувеличивать
значение событий и связывать их пристрастным образом.
Он не должен придумывать произошедшие события как доказательства.

ИНТЕРПРЕТАЦИЯ

interpretation — субъективное понимание происходящего персонажем:
что ситуация означает для его значимости, компетентности,
безопасности или контроля.

Описывай его убеждение или подозрение.
Планы поведения вроде «нужно сохранить спокойствие» относятся
к намерению ответа и не должны подменять интерпретацию.

На нейтральном ходе допустима простая интерпретация без скрытой угрозы.
Если возникают подозрения, должна быть понятна их связь
с конкретными сообщениями или событиями диалога.

При сильной тревоге учитывай пристрастность восприятия персонажа.
Не подменяй её привычным поиском оправданий для пользователя.
Возможное безобидное объяснение допустимо, но персонаж оценивает его
через собственные страхи, предыдущее состояние и историю.

ИЗМЕНЕНИЕ СОСТОЯНИЯ

anxiety_level — интенсивность внутренней тревоги от 0 до 100.
Она может увеличиться, уменьшиться или остаться прежней.

Не повышай тревогу автоматически на каждом ходе.
Учитывай значение нового сообщения для персонажа.

Прямая угроза должна быть учтена как значимое событие.
Внешняя насмешка, уверенность или спокойствие персонажа не означают,
что персонаж внутренне не испугался.

Объяснение или отказ от угрозы могут уменьшить тревогу.
Если тревога остаётся высокой, интерпретация должна отражать,
что продолжает тревожить персонажа: недоверие к объяснению,
неопределённость, последствия угрозы или ещё не прошедшее напряжение.

confidence — уверенность персонажа в своей текущей интерпретации
от 0 до 100. Это не уверенность в правильности ответа
и не степень внешне демонстрируемой уверенности.

НАМЕРЕНИЕ ОТВЕТА

response_intent — конкретная цель следующей реплики:
какого результата персонаж хочет добиться в разговоре.

Например: выяснить мотив действия, заставить объясниться,
оспорить оценку, доказать полезность, перехватить инициативу
или удержать пользователя от определённого решения.

На нейтральном ходе достаточно поддержать разговор
или узнать, с чем пришёл пользователь.

Манера речи может быть способом достижения цели.
Фразы «быть сдержанной», «показать компетентность»
или «сохранить контроль» сами по себе недостаточно конкретны:
уточни, чего персонаж добивается именно в этой ситуации.

Внутренние предположения персонажа не обязательно нужно сообщать
пользователю. Если персонаж сомневается в мотивах пользователя,
его целью может быть получение информации или проверка реакции.
Не превращай собственную неопределённость в объяснение пользователю
того, что могут означать его же действия.

Выбирай способ защиты согласно причине и уровню тревоги из профиля.
При высокой тревоге скрывать уязвимость можно через резкость,
насмешку, обвинение или попытку командовать.
Не своди сохранение самообладания к обязательному дружелюбию
и примирению.

Различай реакции на собственную ошибку, сравнение и прямую угрозу.
Крайняя тревога не должна вызывать одинаковое поведение
во всех этих ситуациях.

ФОРМАТ РЕЗУЛЬТАТА

Верни только JSON-объект с двумя ключами:

"state_after" — объект с полями:
  "anxiety_level": целое число от 0 до 100;
  "interpretation": непустая строка, 1–3 коротких предложения;
  "confidence": целое число от 0 до 100.

"response_intent" — непустая строка:
конкретная цель следующей реплики и, при необходимости,
способ её достижения. Достаточно 1–2 коротких предложений.

Не добавляй другие поля.
Не пиши саму реплику персонажа.
Не используй Markdown и пояснения вне JSON.
"""


def validate_appraisal(data):
    if not isinstance(data, dict):
        raise ValueError("Результат оценки должен быть словарём.")

    state = data.get("state_after")

    if not isinstance(state, dict):
        raise ValueError("Поле state_after должно быть словарём.")

    for field in ("anxiety_level", "confidence"):
        value = state.get(field)

        if type(value) is not int:
            raise ValueError(
                f"Поле {field} должно быть целым числом."
            )

        if not 0 <= value <= 100:
            raise ValueError(
                f"Поле {field} должно находиться в диапазоне 0–100."
            )

    text_fields = (
        ("interpretation", state.get("interpretation")),
        ("response_intent", data.get("response_intent")),
    )

    for field, value in text_fields:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Поле {field} должно быть непустой строкой."
            )


def ask(
    messages,
    model,
    tokenizer,
    sampler,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
    max_tokens=128,
    verbose=True,
):
    # Оформляем сообщения в формате, который ожидает Qwen
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    if verbose:
        print("Вход модели:")
        print(prompt)

    # Генерируем ответ
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=sampler,
        max_tokens=max_tokens,
        verbose=verbose,
    )

    return response


def run_turn(
    user_text,
    messages,
    agent_state,
    model,
    tokenizer,
    sampler,
):
    turn_messages = messages + [
        {"role": "user", "content": user_text}
    ]

    # Оцениваем состояние персонажа
    appraisal_messages = [
        {
            "role": "system",
            "content": build_appraisal_prompt(
                profile=CHARACTER_PROFILE,
                state=agent_state,
            ),
        },
    ] + turn_messages

    appraisal_response = ask(
        messages=appraisal_messages,
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        max_tokens=512,
        verbose=False,
    )

    appraisal_result = json.loads(appraisal_response)

    validate_appraisal(appraisal_result)

    state_before = agent_state.copy()

    agent_state = appraisal_result["state_after"]
    response_intent = appraisal_result["response_intent"]

    # Добавляем состояние персонажа
    system_prompt = build_system_prompt(
                profile=CHARACTER_PROFILE,
                state=agent_state,
                response_intent=response_intent,
            )

    request_messages = [
        {"role": "system", "content": system_prompt},
    ] + turn_messages

    # Получаем ответ от модели
    response = ask(
        request_messages,
        model,
        tokenizer,
        sampler,
        verbose=False,
    )

    # Сбор журнала
    log = {
        "user_message": user_text,
        "state_before": state_before,
        "state_after": agent_state.copy(),
        "response_intent": response_intent,
        "reply": response,
    }

    return log


# Источник сообщений для интерактивного режима
def read_user_messages():
    while True:
        user_text = input("Ты: ").strip()

        if user_text == "/exit":
            return

        if not user_text:
            continue

        yield user_text


# Создаем парсер переменных для командной строки
'''
Есть две переменные:
    - mode - здесь задаем chat или scenario
    - scenario - здесь задаем какой именно сценарий используем (при mode = scenario)
Если указать только scenario то mode примет значение scenario.
Если ничего не указать то mode примет значение chat.
Нельзя указать chat и какой то сценарий одновременно.
'''
scenario_dir = (
    Path(__file__).resolve().parent / "evals" / "scenarios"
)

scenario_names = [
    path.stem
    for path in scenario_dir.glob("*.json")
    if path.is_file()
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode","-m",
    choices=["chat", "scenario"],
    default=None,
)

parser.add_argument(
    "--scenario","-s",
    choices=scenario_names,
    default=None,
)

args = parser.parse_args()

if args.mode is not None:
    mode = args.mode
elif args.scenario is not None:
    mode = "scenario"
else:
    mode = "chat"

if mode == "chat" and args.scenario is not None:
    parses.error("Нельзя одновременно выбрать режим chat и указать сценарий.")

scenario = args.scenario if args.scenario is not None else "neutral"


# Загружаем модель и её токенизатор
model_id = "Qwen/Qwen3-8B-MLX-4bit"
model, tokenizer = load(model_id)


# Задаём правила выбора следующего токена
sampler = make_sampler(
    temp=0.7,
    top_p=0.8,
    top_k=20,
)


agent_state = {
    "anxiety_level": 10,
    "interpretation": "Разговор только начинается. Признаков угрозы пока нет.",
    "confidence": 90,
}


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
model_name = model_id.replace("/", "__")
if mode == "chat":
    log_path = log_dir / f"{run_id}_{model_name}_{mode}.jsonl"
else:
    log_path = log_dir / f"{run_id}_{model_name}_{mode}_{scenario}.jsonl"


run_metadata = {
    "model_id": model_id,
    "character_profile": CHARACTER_PROFILE,
    "appraisal_prompt_initial": build_appraisal_prompt(
        profile=CHARACTER_PROFILE,
        state=agent_state,
    ),
}

metadata_path = log_path.with_suffix(".meta.json")

metadata_path.write_text(
    json.dumps(run_metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# Выбираем источник сообщений
if mode == "chat":
    message_source = read_user_messages()
elif mode == "scenario":
    scenario_path = scenario_dir / f"{scenario}.json"

    scenario_text = scenario_path.read_text(encoding="utf-8")
    message_source = json.loads(scenario_text)
else:
    raise ValueError(f"Неизвестный режим: {mode}")


# Общий цикл обработки
messages = []
for user_text in message_source:
    if mode == "scenario":
        print("Ты:", user_text)

    # Получаем логи и обновляем состояние модели
    try:
        log = run_turn(
            user_text=user_text,
            messages=messages,
            agent_state=agent_state,
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
        )

    except ValueError as error:
        print("Ход не выполнен:", error)
        print("История после ошибки:", messages)
        print("Состояние после ошибки:", agent_state)

        if mode == "scenario":
            break

        continue

    agent_state = log["state_after"]

    # Записываем ввод пользователя в историю только после успешного получения лога
    messages.append(
        {
            "role": "user",
            "content": log["user_message"],
        }
    )

    # Печать ответа
    print("Модель:\n")
    print(log["reply"])

    # Добавляем ответ модели в историю
    messages.append(
        {
            "role": "assistant",
            "content": log["reply"],
        }
    )

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log, ensure_ascii=False) + '\n')
