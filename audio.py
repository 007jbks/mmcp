import speech_recognition as sr

r = sr.Recognizer()


def convert(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    return text
