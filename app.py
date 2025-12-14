import os
import re
import ssl
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

HF_KEY = os.environ["HUGGINGFACE_API_KEY"]
HF_MODEL_MT = os.environ["HUGGINGFACE_TR_MODEL_URL"]
HF_MODE = os.getenv("HF_MODE", "mt").strip().lower()  # "mt" | "llm"
HF_LLM_MODEL = os.getenv("HUGGINGFACE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_LLM_BASE_URL = os.getenv("HF_LLM_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")

app = FastAPI()

SRT_TIMESTAMP_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}(\s+.*)?\s*$"
)


class TLSHttpAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        pool_kwargs["ssl_context"] = ctx
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **pool_kwargs)


session = requests.Session()
session.mount("https://", TLSHttpAdapter())

openai_client = None
if HF_MODE == "llm":
    openai_client = OpenAI(api_key=HF_KEY, base_url=HF_LLM_BASE_URL)
log.info("Translator init: mode=%s mt_model=%s llm_model=%s llm_base=%s", HF_MODE, HF_MODEL_MT, HF_LLM_MODEL, HF_LLM_BASE_URL)


@app.get("/health")
def health():
    return {"status": "ok"}


def _rewrite_hf_url_if_410(url: str) -> str | None:
    old_prefix = "https://api-inference.huggingface.co/models/"
    if not url.startswith(old_prefix):
        return None
    model_id = url.removeprefix(old_prefix)
    return f"https://router.huggingface.co/hf-inference/models/{model_id}"


def _extract_hf_text(data) -> str:
    if isinstance(data, list) and data:
        data = data[0]

    if isinstance(data, dict):
        if "error" in data:
            raise HTTPException(status_code=502, detail=str(data["error"]))
        if "translation_text" in data:
            return str(data["translation_text"])
        if "generated_text" in data:
            return str(data["generated_text"])

    if isinstance(data, str):
        return data

    raise HTTPException(status_code=502, detail=f"Unexpected HF response: {type(data).__name__}")


def _contains_cyrillic(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch == "ё" for ch in text)


def _translate_line_mt(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_KEY}",
        "Content-Type": "application/json",
    }

    body = {"inputs": text}

    resp = session.post(HF_MODEL_MT, headers=headers, json=body, timeout=180)
    if resp.status_code == 410:
        new_url = _rewrite_hf_url_if_410(HF_MODEL_MT)
        if new_url:
            resp = session.post(new_url, headers=headers, json=body, timeout=180)
    resp.raise_for_status()
    return _extract_hf_text(resp.json()).strip()


def _translate_line_en_ru(text: str) -> str:
    llm_result = None
    if HF_MODE == "llm":
        try:
            resp = openai_client.chat.completions.create(
                model=HF_LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translation engine. "
                            "Translate from English to Russian. "
                            "Output ONLY the Russian translation. "
                            "Do not repeat the source, do not add quotes or comments."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM translation failed: {e}") from e

        if not resp.choices:
            raise HTTPException(status_code=502, detail="Empty response from LLM")
        content = resp.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Empty message from LLM")
        llm_result = content.strip()
        if llm_result and _contains_cyrillic(llm_result) and llm_result.lower() != text.strip().lower():
            log.info("LLM used")
            return llm_result
        log.warning("LLM returned non-cyrillic/identical text, will try MT: %s -> %s", text, llm_result)

    mt_result = _translate_line_mt(text)
    if mt_result:
        log.info("MT used")
    return mt_result


@app.post("/translate")
async def translate(request: Request):
    srt_text = (await request.body()).decode("utf-8", errors="ignore")
    out_lines: list[str] = []
    for line in srt_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or SRT_TIMESTAMP_RE.match(line):
            out_lines.append(line)
            continue
        try:
            out_lines.append(_translate_line_en_ru(line))
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"HuggingFace request failed: {e}") from e

    return Response(content="\n".join(out_lines) + "\n", media_type="text/plain")
