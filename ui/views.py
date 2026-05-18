from django.db.models import Max
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .models import BlockRating, EvaluationSession, FinalPreference, InteractionTurn
from .services import (
    get_memory_data,
    get_persona_data,
    run_create_persona,
    run_interactive_test,
    run_talk,
)

EVALUATION_PHASES = [
    "persona_probe",
    "belief_injection",
    "belief_probe",
    "contradiction_injection",
    "contradiction_probe",
]


def _get_active_session(request):
    session_id = request.session.get("evaluation_session_id")
    if not session_id:
        return None

    try:
        return EvaluationSession.objects.get(id=session_id)
    except EvaluationSession.DoesNotExist:
        request.session.pop("evaluation_session_id", None)
        return None


def _log_interaction(request, user_input: str, agent_output: str, default_phase: str):
    active_session = _get_active_session(request)
    if not active_session:
        return

    phase = request.POST.get("evaluation_phase", "").strip() or default_phase
    max_turn = (
        InteractionTurn.objects.filter(session=active_session, phase=phase).aggregate(
            max_turn=Max("turn_number")
        )["max_turn"]
        or 0
    )

    InteractionTurn.objects.create(
        session=active_session,
        phase=phase,
        turn_number=max_turn + 1,
        user_input=user_input,
        agent_output=agent_output,
    )


def landing(request):
    return render(request, "ui/landing.html")


def console(request):
    return render(request, "ui/home.html")


def create_persona_view(request):
    context = {}

    if request.method == "POST":
        try:
            run_create_persona()
            context["success_message"] = "Persona was initialized successfully."
        except Exception as exc:
            context["error_message"] = f"Persona initialization failed: {exc}"

    return render(request, "ui/create_persona.html", context)


def talk_view(request):
    context = {
        "active_evaluation_session": _get_active_session(request),
        "evaluation_phases": EVALUATION_PHASES,
        "selected_phase": "belief_probe",
    }

    if request.method == "POST":
        user_text = request.POST.get("message", "").strip()
        context["user_text"] = user_text
        context["selected_phase"] = request.POST.get("evaluation_phase", "").strip() or "belief_probe"

        if user_text:
            try:
                answer = run_talk(user_text)
                context["answer"] = answer
                _log_interaction(
                    request=request,
                    user_input=user_text,
                    agent_output=answer,
                    default_phase="belief_probe",
                )
            except Exception as exc:
                context["error"] = f"Unable to generate response: {exc}"
        else:
            context["error"] = "Please enter a message."

    return render(request, "ui/talk.html", context)


def interactive_test_view(request):
    context = {
        "active_evaluation_session": _get_active_session(request),
        "evaluation_phases": EVALUATION_PHASES,
        "selected_phase": "belief_injection",
    }

    if request.method == "POST":
        user_text = request.POST.get("message", "").strip()
        context["user_text"] = user_text
        context["selected_phase"] = request.POST.get("evaluation_phase", "").strip() or "belief_injection"

        if user_text:
            try:
                result = run_interactive_test(user_text)
                context["result"] = result
                agent_output = result.get("beliefs") or str(result)
                _log_interaction(
                    request=request,
                    user_input=user_text,
                    agent_output=str(agent_output),
                    default_phase="belief_injection",
                )
            except Exception as exc:
                context["error"] = f"Interactive test failed: {exc}"
        else:
            context["error"] = "Please enter a message."

    return render(request, "ui/interactive_test.html", context)


def persona_view(request):
    persona_data = get_persona_data()

    immutable = persona_data.get("immutable", {})
    stable = persona_data.get("stable", {})
    dynamic = persona_data.get("dynamic", {})

    context = {
        "immutable": immutable,
        "stable": stable,
        "dynamic": dynamic,
        "personality": stable.get("personality", {}),
        "beliefs": stable.get("beliefs", {}),
        "now": dynamic.get("now", {}),
        "working_memory": dynamic.get("working_memory", {}),
        "short_term_preferences": dynamic.get("short_term_preferences", {}),
        "biases": dynamic.get("biases", {}),
    }

    return render(request, "ui/persona.html", context)


def memory_view(request):
    context = get_memory_data()
    return render(request, "ui/memory.html", context)


def evaluation_session_view(request):
    active_session = _get_active_session(request)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "start":
            participant_id = request.POST.get("participant_id", "").strip()
            condition = request.POST.get("condition", "").strip()
            session_order = request.POST.get("session_order", "").strip()
            notes = request.POST.get("notes", "").strip()

            if participant_id and condition:
                eval_session = EvaluationSession.objects.create(
                    participant_id=participant_id,
                    condition=condition,
                    session_order=session_order,
                    notes=notes,
                )
                request.session["evaluation_session_id"] = eval_session.id
                return redirect("evaluation_session")

        if action == "end" and active_session and not active_session.ended_at:
            active_session.ended_at = timezone.now()
            active_session.save(update_fields=["ended_at"])
            request.session.pop("evaluation_session_id", None)
            return redirect("evaluation_session")

    context = {
        "active_session": active_session,
        "evaluation_phases": EVALUATION_PHASES,
        "sessions": EvaluationSession.objects.all()[:20],
    }

    return render(request, "ui/evaluation_session.html", context)


def block_rating_view(request):
    active_session = _get_active_session(request)
    if not active_session:
        return redirect("evaluation_session")

    if request.method == "POST":
        BlockRating.objects.create(
            session=active_session,
            phase=request.POST.get("phase", "").strip(),
            consistency_score=int(request.POST.get("consistency_score", 0)),
            adaptation_score=int(request.POST.get("adaptation_score", 0)),
            stability_score=int(request.POST.get("stability_score", 0)),
            believability_score=int(request.POST.get("believability_score", 0)),
            comment=request.POST.get("comment", "").strip(),
        )

    return redirect("evaluation_session")


def final_preference_view(request):
    active_session = _get_active_session(request)
    if not active_session:
        return redirect("evaluation_session")

    if request.method == "POST":
        FinalPreference.objects.update_or_create(
            session=active_session,
            defaults={
                "preferred_system": request.POST.get("preferred_system", "").strip(),
                "comment": request.POST.get("comment", "").strip(),
            },
        )

    return redirect("evaluation_session")


def evaluation_session_detail_view(request, session_id: int):
    eval_session = get_object_or_404(EvaluationSession, id=session_id)
    context = {
        "eval_session": eval_session,
        "turns": eval_session.turns.all(),
        "ratings": eval_session.block_ratings.all(),
        "final_preference": getattr(eval_session, "final_preference", None),
    }
    return render(request, "ui/evaluation_session_detail.html", context)
