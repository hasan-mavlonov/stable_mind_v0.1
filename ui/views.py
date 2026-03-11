from django.shortcuts import render
from .services import (
    run_create_persona,
    run_talk,
    get_persona_data,
    get_memory_data,
    run_interactive_test,
)


def home(request):
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
    context = {}

    if request.method == "POST":
        user_text = request.POST.get("message", "").strip()
        context["user_text"] = user_text

        if user_text:
            try:
                context["answer"] = run_talk(user_text)
            except Exception as exc:
                context["error"] = f"Unable to generate response: {exc}"
        else:
            context["error"] = "Please enter a message."

    return render(request, "ui/talk.html", context)


def interactive_test_view(request):
    context = {}

    if request.method == "POST":
        user_text = request.POST.get("message", "").strip()
        context["user_text"] = user_text

        if user_text:
            try:
                result = run_interactive_test(user_text)
                context["result"] = result
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
