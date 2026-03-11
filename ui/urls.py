from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("create-persona/", views.create_persona_view, name="create_persona"),
    path("talk/", views.talk_view, name="talk"),
    path("interactive-test/", views.interactive_test_view, name="interactive_test"),
    path("persona/", views.persona_view, name="persona"),
    path("memory/", views.memory_view, name="memory"),
    path("evaluation/", views.evaluation_session_view, name="evaluation_session"),
    path("evaluation/rating/", views.block_rating_view, name="block_rating"),
    path("evaluation/final-preference/", views.final_preference_view, name="final_preference"),
    path("evaluation/<int:session_id>/", views.evaluation_session_detail_view, name="evaluation_session_detail"),
]
