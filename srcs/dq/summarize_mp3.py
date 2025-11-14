import tempfile
import ffmpeg
import json
import os
from vosk import Model, KaldiRecognizer
from agent import call_agent, prompt


VOSK_MODEL_PATH = "vosk-model-small-cs-0.4-rhasspy"
model = Model(VOSK_MODEL_PATH)


def convert_to_wav(input_path: str, sample_rate: int = 16000) -> str:
    """
    Convert an input audio file to a temporary WAV file.

    Parameters
    ----------
    input_path : str
        Path to the original audio file.
    sample_rate : int
        Target sample rate for Vosk processing.

    Returns
    -------
    str
        Path to the generated temporary WAV file.
    """
    fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Convert using ffmpeg with mono channel and target sample rate
    (
        ffmpeg
        .input(input_path)
        .output(tmp_wav_path, ac=1, ar=sample_rate, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    return tmp_wav_path


def transcribe_audio(path: str) -> str:
    """
    Transcribe an audio file using the Vosk model.

    Parameters
    ----------
    path : str
        Path to the audio file to transcribe.

    Returns
    -------
    str
        Raw transcript text extracted by Vosk.
    """
    wav_path = convert_to_wav(path)

    try:
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)

        transcript_chunks = []

        # Stream WAV file to Vosk in chunks
        with open(wav_path, "rb") as wf:
            while True:
                data = wf.read(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    transcript_chunks.append(res.get("text", ""))

            # Include final recognition output
            final_res = json.loads(rec.FinalResult())
            transcript_chunks.append(final_res.get("text", ""))

        return " ".join(transcript_chunks).strip()

    finally:
        # Cleanup temporary WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)


def clean_czech_text(text: str) -> str:
    """
    Remove Vosk placeholder artifacts from Czech transcripts.

    Parameters
    ----------
    text : str
        Raw transcript.

    Returns
    -------
    str
        Cleaned transcript with filler tokens removed.
    """
    return text.replace("[unk]", "").strip()


def transform_to_summary(path: str):
    """
    Convert an audio recording into a structured summary by:
    1. Transcribing audio with Vosk
    2. Cleaning transcription artifacts
    3. Sending the cleaned text to the Llama model for summarization

    Parameters
    ----------
    path : str
        Path to the input audio file.

    Returns
    -------
    str
        Summary produced by the language model.
    """
    transcript = transcribe_audio(path)
    cleaned = clean_czech_text(transcript)
    summary = call_agent(prompt, cleaned)
    return summary
