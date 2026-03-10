from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("create-persona/", views.create_persona_view, name="create_persona"),
    path("talk/", views.talk_view, name="talk"),
    path("interactive-test/", views.interactive_test_view, name="interactive_test"),
    path("persona/", views.persona_view, name="persona"),
    path("memory/", views.memory_view, name="memory"),
]