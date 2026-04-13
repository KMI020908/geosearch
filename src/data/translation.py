"""LLM-based geographic name translation and Latin-to-Cyrillic correction utilities."""
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# API key is read from the environment (set in .env / shell).
_DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")

geo_translator = ChatOpenAI(
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    api_key=_DEEPSEEK_API_KEY,
    temperature=0,
    top_p=0.9,
    max_tokens=20,
    extra_body={
        "think": "minimal",
        "chat_template_kwargs": {"enable_thinking": False},
    },
    default_headers={"Authorization": f"Bearer {_DEEPSEEK_API_KEY}"},
    seed=67,
)

_SYSTEM_PROMPT = """\
You are a professional geographic name translator.

Rules:
1. You ONLY translate geographic entities (cities, countries, rivers, regions, landmarks).
2. Target language is provided in the user prompt.
3. If a well-known translation exists → use it.
4. If you are NOT sure about the translation → return a transliteration instead.
5. NEVER hallucinate or invent names.
6. Output ONLY the final answer, no explanations.
7. No additional text.

Definition:
- Translation = commonly used localized name
- Transliteration = phonetic conversion to target language

Examples:
Input: "Germany", target: ru
Output: Германия

Input: "Mississippi River", target: ru
Output: Миссисипи

Input: "UnknownSmallVillageX", target: ru
Output: АнкноунСмолВилладжИкс
"""

_USER_PROMPT_TEMPLATE = """\
Translate the following geographic name into {target_lang}.

Text: {text}

Remember:
- If translation is known -> translate
- If not -> transliterate
"""


def build_messages(text: str, target_lang: str) -> list:
    """Build a LangChain message list for translating one geographic name."""
    prompt = _USER_PROMPT_TEMPLATE.format(text=text, target_lang=target_lang)
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]


# ── Latin → Cyrillic lookalike substitution ───────────────────────────────────
# Some OCR pipelines or transliterations produce Cyrillic words written with
# visually identical Latin characters.  This corrects them while leaving
# Roman numerals (which legitimately use Latin letters) untouched.

_LAT_TO_CYR = str.maketrans(
    "AaBCcEeHIKkMOoPpTXxYy",
    "АаВСсЕеНІКкМОоРрТХхУу",
)

_ROMAN_RE = re.compile(
    r"\b(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})"
    r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b"
)


def latin_to_cyrillic(text: str) -> str:
    """
    Replace Latin lookalike characters with their Cyrillic counterparts.
    Roman numerals are detected first and restored unchanged after substitution.
    """
    romans: list[str] = []
    counter = [0]

    def _save(m: re.Match) -> str:
        romans.append(m.group())
        idx = counter[0]
        counter[0] += 1
        return f"\x00{idx}\x00"

    masked = _ROMAN_RE.sub(_save, text).translate(_LAT_TO_CYR)
    return re.sub(r"\x00(\d+)\x00", lambda m: romans[int(m.group(1))], masked)
