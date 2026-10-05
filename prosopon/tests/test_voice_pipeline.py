"""Unit tests for local voice transcription and speech synthesis."""

from pathlib import Path
import pytest

from prosopon.voice.transcoder import transcode_to_wav, AudioTranscodingError
from prosopon.voice.whisper_stt import WhisperSTTEngine, STTUnavailableError
from prosopon.voice.piper_tts import PiperTTSEngine, TTSUnavailableError


@pytest.mark.asyncio
async def test_transcoder_empty_input_rejected():
    with pytest.raises(AudioTranscodingError):
        await transcode_to_wav(b"")


@pytest.mark.asyncio
async def test_transcoder_valid_audio():
    import io, struct, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        frames = struct.pack("<1000h", *([0] * 1000))
        wf.writeframes(frames)
    
    out_wav = await transcode_to_wav(buf.getvalue())
    assert out_wav.exists()
    assert out_wav.suffix == ".wav"
    out_wav.unlink()


@pytest.mark.asyncio
async def test_whisper_stt_fails_honestly_when_binary_missing(tmp_path):
    stt = WhisperSTTEngine(binary_name="nonexistent_whisper_binary", mock_mode=False)
    mock_wav = tmp_path / "test.wav"
    mock_wav.write_bytes(b"RIFF....WAVEfmt ....data....")

    with pytest.raises(STTUnavailableError, match="not found on PATH"):
        await stt.transcribe(mock_wav)


@pytest.mark.asyncio
async def test_whisper_stt_mock_mode_returns_simulated_provenance(tmp_path):
    stt = WhisperSTTEngine(binary_name="nonexistent_whisper_binary", mock_mode=True)
    mock_wav = tmp_path / "test.wav"
    mock_wav.write_bytes(b"RIFF....WAVEfmt ....data....")

    text, duration_ms, is_sim = await stt.transcribe(mock_wav)
    assert "[SIMULATED TRANSCRIPT]" in text
    assert is_sim is True
    assert duration_ms >= 0.0


@pytest.mark.asyncio
async def test_piper_tts_fails_honestly_when_binary_missing():
    tts = PiperTTSEngine(binary_name="nonexistent_piper_binary", mock_mode=False)
    with pytest.raises(TTSUnavailableError, match="not found on PATH"):
        await tts.synthesize("Hello from Metis")


@pytest.mark.asyncio
async def test_piper_tts_mock_mode_returns_simulated_provenance():
    tts = PiperTTSEngine(binary_name="nonexistent_piper_binary", mock_mode=True)
    wav_bytes, duration_ms, is_sim = await tts.synthesize("Hello from Metis")
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0
    assert is_sim is True
    assert duration_ms >= 0.0

