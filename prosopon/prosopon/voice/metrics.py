"""Voice round-trip latency instrumentation and phase metrics."""

from pydantic import BaseModel, Field


class VoicePhaseMetrics(BaseModel):
    """Timing breakdown across the push-to-talk voice pipeline."""
    mic_stop_to_transcript_ms: float = Field(0.0, description="Duration from mic release to completed STT transcript")
    transcript_to_first_text_ms: float = Field(0.0, description="Duration from transcript submission to first response token")
    answer_completion_ms: float = Field(0.0, description="Duration to complete full text synthesis")
    first_audio_ms: float = Field(0.0, description="Duration to synthesize first TTS audio chunk")
    total_round_trip_ms: float = Field(0.0, description="Total end-to-end voice turn duration")
