PRODUCT REQUIREMENTS DOCUMENT – GEMINI RAG PLAYGROUND

VERSION: 0.1  DATE: 19 Jun 2025
AUTHOR: ChatGPT draft for Kanu

PURPOSE & SCOPE

Build a self‑contained Retrieval‑Augmented Generation (RAG) playground that:
• Works without Looker credentials by default (using a stub)
• Uses Google Gemini for both text generation and embeddings
• Imitates the key Looker flow so that the real Looker SDK can be dropped in later with zero code changes
• Runs locally in < 5 minutes so it can be executed by a Cursor agent, a notebook, or a CI job

USERS & USE‑CASES

U1 (Data Engineer) Tests prompt design and JSON‑schema validation offline.
U2 (Analytics Engineer) Verifies that a planned LookML model will answer plain‑English questions before hitting production.
U3 (LLM Engineer) Integrates the loop into a multistep agent (e.g. Crew AI or Cursor agent) and later replaces the stub with the real Looker SDK.

FUNCTIONAL REQUIREMENTS

FR‑1 Generate a synthetic orders fact table with ≥1 000 rows (numeric + dates).
FR‑2 Emit lightweight “field documents” (one file per dimension/measure) for embedding.
FR‑3 Create a FAISS vector index using Gemini embeddings.
FR‑4 Accept a natural‑language question, retrieve top‑k field docs, and ask Gemini to return a valid JSON WriteQuery object.
FR‑5 Execute the JSON query against the synthetic dataframe (or later the real Looker API) and return results.
FR‑6 Enforce a default row limit to avoid accidental large dumps.
FR‑7 All code runnable by:    python make_synthetic_data.py → python build_index.py → python chat_gemini.py

NON‑FUNCTIONAL REQUIREMENTS

NFR‑1 Local‑only I/O except one outbound call to Gemini endpoints.
NFR‑2 Setup in < 2 min on a modern laptop (M‑series or x86).
NFR‑3 Round‑trip latency (retrieve ➔ generate ➔ execute) < 1 s for a 1 000‑row dataset using gemini‑2.5‑flash.
NFR‑4 All dependencies managed via pip; no Docker required.
NFR‑5 Source code must be plaintext so Cursor agent can parse/refactor.

SYSTEM DESIGN

COMPONENTS
· make_synthetic_data.py Creates CSV and field docs → fake_metadata.json
· fake_looker.py Minimal stub exposing the three SDK methods our loop needs
· build_index.py Embeds docs with Gemini‑embedding‑001 and saves FAISS index
· chat_gemini.py End‑to‑end RAG loop (retrieve → Gemini Flash → execute → return)

DATA FLOW
1 User question ➔ chat_gemini.py
2 Script pulls k nearest docs from FAISS.
3 Docs + question ➔ Gemini Flash ➔ JSON WriteQuery.
4 JSON ➔ fake_looker.run_inline_query ➔ pandas filter ➔ rows.
5 Rows printed / returned to agent.

SETUP & RUNBOOK

Prereqs: Python ≥ 3.10, internet to reach api.gemini.

export GOOGLE_API_KEY="<your‑key>"

pip install -r requirements.txt (see below)

python make_synthetic_data.py

python build_index.py

python chat_gemini.py – ask your first question.

requirements.txt

pandas
faiss-cpu
google-generativeai
llama-index
llama-index-embeddings-google-genai

FOLDER STRUCTURE (checked into repo root)

rag_playground/
├── fake_looker.py
├── make_synthetic_data.py
├── build_index.py
├── chat_gemini.py
├── lookml_docs/                # auto‑generated docs per field
├── orders.csv                  # auto‑generated fact table
├── fake_metadata.json          # auto‑generated LookML stub
└── lookml_faiss/               # persisted FAISS index

FULL SOURCE LISTINGS

make_synthetic_data.py

import pandas as pd, random, pathlib, json, datetime as dt

BASE = pathlib.Path('.')
DOCS = BASE / 'lookml_docs'
DOCS.mkdir(exist_ok=True)

MODEL = 'ecommerce'
EXPLORE = 'orders'
fields = [
    {'name': 'orders.id',           'label': 'Order ID',      'type': 'dimension', 'sql': '${TABLE}.id'},
    {'name': 'orders.created_date', 'label': 'Created Date',  'type': 'dimension', 'sql': '${TABLE}.created_date'},
    {'name': 'orders.total_sale',   'label': 'Total Sale $',  'type': 'measure',   'sql': 'SUM(${TABLE}.sale)'},
]

meta = {'model': MODEL, 'explore': EXPLORE, 'fields': fields}
(BASE / 'fake_metadata.json').write_text(json.dumps(meta, indent=2))

for f in fields:
    text = f"{f['name']}\n{f['label']}\n{f['sql']}"
    (DOCS / f"{f['name']}.txt").write_text(text)

N = 1_000
today = dt.date.today()
data = {
    'id': range(1, N + 1),
    'created_date': [today - dt.timedelta(days=random.randint(0, 60)) for _ in range(N)],
    'sale': [round(random.uniform(20, 500), 2) for _ in range(N)],
}

pd.DataFrame(data).to_csv(BASE / 'orders.csv', index=False)
print('🟢  Synthetic data + metadata ready.')

fake_looker.py

import pandas as pd, json
_meta = json.loads(open('fake_metadata.json').read())
_df   = pd.read_csv('orders.csv', parse_dates=['created_date'])

def all_lookml_models(fields=None):
    return [{'name': _meta['model'], 'label': _meta['model'].title()}]

def lookml_model(model_name, fields=None):
    if model_name != _meta['model']:
        raise ValueError('unknown model')
    return {'name': model_name, 'explores': [{'name': _meta['explore'], 'label': _meta['explore'].title()}]}

def run_inline_query(result_format, query):
    if query['model'] != _meta['model'] or query['view'] != _meta['model']:
        raise ValueError('bad model/view')
    df = _df.copy()
    for key, val in (query.get('filters') or {}).items():
        if key == 'orders.created_date' and 'days' in val:
            days = int(val.split()[0])
            cutoff = _df['created_date'].max() - pd.Timedelta(days=days)
            df = df[df['created_date'] >= cutoff]
    if query.get('limit'):
        df = df.head(int(query['limit']))
    if result_format == 'json':
        cols = [c.split('.')[1] for c in query['fields']]
        return df[cols].to_json(orient='records')
    raise NotImplementedError

build_index.py (Gemini embeddings)

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

docs = SimpleDirectoryReader('lookml_docs').load_data()
index = VectorStoreIndex.from_documents(
    docs,
    embed_model=GoogleGenAIEmbedding(model_name='gemini-embedding-001'),
    show_progress=True
)
index.storage_context.persist('lookml_faiss')
print('🟢  Vector index built with Gemini embeddings.')

chat_gemini.py

import os, json, fake_looker as sdk
from llama_index import VectorStoreIndex
from google import generativeai as genai

os.environ['GOOGLE_API_KEY']  # ensure set externally
index = VectorStoreIndex.load_from_disk('lookml_faiss')
retriever = index.as_retriever(similarity_top_k=4)
client = genai.Client()
MODEL_ID = 'gemini-2.5-flash'

def ask(question: str, limit: int = 10):
    ctx = '\n\n'.join(n.node.get_content() for n in retriever.retrieve(question))
    prompt = f"""{ctx}\n\nUser: {question}\nReturn ONLY JSON with keys model, view, fields, filters, limit."""
    resp = client.models.generate_content(model=MODEL_ID, contents=prompt, stream=False)
    query = json.loads(resp.text.strip())
    query.setdefault('limit', limit)
    rows = json.loads(sdk.run_inline_query('json', query))
    return rows

if __name__ == '__main__':
    print(ask('show total_sale and id for the last 30 days limit 5'))

9. FUTURE ROADMAP

• Swap fake_looker.py for from looker_sdk import init40 once credentials work.
• Add Gemini function‑calling schema for strict JSON enforcement.
• Implement streaming output for conversation‑style UI.
• Add integration test that validates JSON against pydantic model.

10. REFERENCES

Gemini docs https://ai.google.devFAISS https://github.com/facebookresearch/faissLlama‑Index https://github.com/run-llama/llama_index

=== END OF DOCUMENT ===