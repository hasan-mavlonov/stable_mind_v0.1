from django.contrib import admin

from .models import BlockRating, EvaluationSession, FinalPreference, InteractionTurn


admin.site.register(EvaluationSession)
admin.site.register(InteractionTurn)
admin.site.register(BlockRating)
admin.site.register(FinalPreference)
