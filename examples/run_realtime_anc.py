from src.realtime.live_audio_filter import LiveAudioFilter


FILTER_ORDER = 64

VSNLMS_PARAMS = (
    0.0001,
    0.1,
    0.000001,
    10,
    10,
    1.01,
)

WEIGHTS_PATH = "adaptive_filter_weights.npy"


runtime_filter = LiveAudioFilter(
    FILTER_ORDER,
    VSNLMS_PARAMS,
    WEIGHTS_PATH,
)


runtime_filter.start(
    duration=10,
    temp_filename="temp_recording.wav",
    output_filename="filtered_output.wav",
)
