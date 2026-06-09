# MindForm

**The personality layer for AI agents.**

> Memory alone isn't enough. MindForm gives intelligent agents **behavioral
> continuity**, **emotional persistence**, and an **evolving identity** that
> stays recognizably itself across thousands of interactions.

MindForm is a memory- and reflection-based architecture that decouples
short-term *belief plasticity* from long-term *trait evolution*. An agent's
view of the world adapts fast; who the agent *is* changes slowly, only when
evidence is overwhelming. Perception and language run on **Gemma** open
models via the Google AI ecosystem.

![MindForm Architecture](assets/diagrams/mindform_architecture.svg)

---

## Why MindForm

Standard LLM memory remembers *facts*. It does not preserve *identity*.
Without a stable self, agents drift — sycophantic, inconsistent, generic.
MindForm adds the layer that decides what new information *means* for who
the agent is:

- **Personality Engine** — a persistent, auditable trait vector that sits
  above the model (model-agnostic).
  - **Behavioral Continuity** — convictions survive thousands of
    conversations instead of dissolving into agreement.
  - **Emotional Persistence** — mood, stress, and energy carry across
    sessions and color responses without overwriting the core self.
  - **Long-Term Identity** — two clocks (fast beliefs, slow traits) with
    bounded reflection so personality matures without snapping.

---

## Gemma / Google AI integration

MindForm runs entirely on Google's **Gemma 4 (31B IT)** model via the Google AI ecosystem.

- All LLM access is centralized in [`core/llm.py`](core/llm.py) using the official `google-genai` SDK.

- The **Perception Engine** ([`core/perception.py`](core/perception.py)) uses Gemma 4 to extract structured, ontology-constrained perceptions in JSON format.

- The **Talk Interface** ([`talk.py`](talk.py)) uses Gemma 4 to generate personality-consistent responses.

- The **Reflection System** ([`core/reflection.py`](core/reflection.py)) uses Gemma 4 for long-term personality trait evolution.

The default model is:

gemma-4-31b-it


All model behavior is controlled via environment variables.

---

## Environment variables

Copy [`.env.example`](.env.example) to `.env` and fill in your key:

```env
# Gemma / Google AI ecosystem (GEMMA_API_KEY is primary; the others are fallbacks)
GEMMA_API_KEY=
GOOGLE_API_KEY=
GOOGLE_AI_STUDIO_KEY=

# Model selection (any Gemma model on the Gemini API)
MINDFORM_MODEL=gemma-4-31b-it
TALK_MODEL=gemma-4-31b-it
PERCEPTION_MODEL=gemma-4-31b-it

# Django
DJANGO_SECRET_KEY=
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=*
```

| Variable | Purpose |
| --- | --- |
| `GEMMA_API_KEY` | Primary Gemma / Google AI Studio key |
| `GOOGLE_API_KEY` / `GOOGLE_AI_STUDIO_KEY` | Accepted fallbacks |
| `MINDFORM_MODEL` | Default Gemma model for all engines |
| `TALK_MODEL` | Override model for the Talk interface |
| `PERCEPTION_MODEL` | Override model for perception extraction |
| `DJANGO_SECRET_KEY` | Django secret key (set in production) |
| `DJANGO_DEBUG` | `True`/`False` |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated hosts (`*` for dev) |

---

## Run locally

Requires Python 3.11+.

```bash
# 1. Clone and enter the project
cd stable_mind_v0.1

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your environment
cp .env.example .env             # then add your GEMMA_API_KEY

# 5. Apply database migrations
python manage.py migrate

# 6. Run the development server
python manage.py runserver
```

Open <http://127.0.0.1:8000/>:

- `/` — the **MindForm landing site** (futuristic, animated overview).
  - `/console/` — the **MindForm Console** to drive the live engine.

### CLI tools

```bash
python create_persona.py --root .     # initialize a fresh persona
python talk.py                        # read-only conversational interface
python interactive_test.py            # run single engine steps
```

---

## Project structure

```
core/                Personality engine
  llm.py             Single entry point to the Gemma / Google AI ecosystem
  perception.py      Gemma-powered structured perception extraction
  agent.py           Orchestrates perception → belief → reflection
  consolidation.py   Belief consolidation (fast loop)
  reflection.py      Bounded trait evolution (slow loop)
  state_manager.py   Persona / state persistence
talk.py              Read-only "talk to the persona" interface (Gemma)
interactive_test.py  Single-step engine CLI
create_persona.py    Persona / memory bootstrap
ui/                   Django app: landing site + console
  templates/ui/landing.html   Cinematic MindForm landing page
  templates/ui/*.html         Console (persona, talk, memory, evaluation)
webapp/               Django project (env-driven settings)
persona/ memory/ state/ config/   Engine data
```

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for details.
