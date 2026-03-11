from django.db import models


class EvaluationSession(models.Model):
    participant_id = models.CharField(max_length=128)
    condition = models.CharField(max_length=64)
    session_order = models.CharField(max_length=64, blank=True)
    notes = models.TextField(blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"{self.participant_id} ({self.condition})"


class InteractionTurn(models.Model):
    session = models.ForeignKey(EvaluationSession, on_delete=models.CASCADE, related_name="turns")
    phase = models.CharField(max_length=64)
    turn_number = models.PositiveIntegerField()
    user_input = models.TextField()
    agent_output = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]


class BlockRating(models.Model):
    session = models.ForeignKey(EvaluationSession, on_delete=models.CASCADE, related_name="block_ratings")
    phase = models.CharField(max_length=64)
    consistency_score = models.IntegerField()
    adaptation_score = models.IntegerField()
    stability_score = models.IntegerField()
    believability_score = models.IntegerField()
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]


class FinalPreference(models.Model):
    session = models.OneToOneField(EvaluationSession, on_delete=models.CASCADE, related_name="final_preference")
    preferred_system = models.CharField(max_length=64)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
