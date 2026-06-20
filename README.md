# Speech to Text with Automated Content Filtering

A Streamlit app that records speech from your microphone, transcribes it to text, and automatically censors profanity and negative language in real time.

## Features
- Live mic recording with timer
- Speech-to-text via Google Speech Recognition
- Automatic filtering of profanity and negative words
- Word/blocked stats + recording history
- Dark/light mode toggle

## Requirements
- Python 3.12
- Microphone + internet connection

## Installation
```bash
py -3.12 -m pip install streamlit SpeechRecognition sounddevice scipy better-profanity
```

## Run
```bash
py -3.12 -m streamlit run speech_to_text_filter.py
```
