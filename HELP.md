Pipecat - open-source Python framework for building real-time voice and multimodal AI agents.

## Flow

1. User microphone - Take voice from microphone - taken using telephony 

Pipecat WebRTC : Raw Audio Ingestion

- Capture audio from the browser microphone, transport it to Python using WebRTC, convert the incoming audio into Pipecat Frames, and inspect those frames.

Transport - how audio reaches Python
Frames - how Pipecat represents audio data
Pipeline - how data flows between processors
FrameProcessor - how we inspect or manipulate frames