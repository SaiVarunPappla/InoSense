import speech_recognition as sr

r = sr.Recognizer()

print("Testing microphone... Say something!")
with sr.Microphone() as source:
    # Adjust for noise first
    r.adjust_for_ambient_noise(source, duration=2)
    print("Speak now (say 'hello testing'):")
    audio = r.listen(source, timeout=5)

try:
    text = r.recognize_google(audio)
    print("You said:", text)
except sr.UnknownValueError:
    print("Sorry, I couldn't understand you.")
except sr.RequestError:
    print("Could not connect to Google's servers. Check your internet.")