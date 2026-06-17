import speech_recognition as sr
import pyttsx3
from deep_translator import GoogleTranslator

# -----------------------------------
# Initialize text-to-speech engine
# -----------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Speech recognizer
recognizer = sr.Recognizer()


# -----------------------------------
# Speak text
# -----------------------------------
def speak(text):
    print("Speaking:", text)
    engine.say(text)
    engine.runAndWait()


# -----------------------------------
# Convert speech to text
# -----------------------------------
def speech_to_text():
    with sr.Microphone() as source:
        print("Please speak now in English...")

        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=10
            )
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return ""

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio, language="en-US")
        print("You said:", text)
        return text

    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print("API Error:", e)

    return ""


# -----------------------------------
# Translate text
# -----------------------------------
def translate_text(text, target_language):
    try:
        translated = GoogleTranslator(
            source="en",
            target=target_language
        ).translate(text)

        print("Translated text:", translated)
        return translated

    except Exception as e:
        print("Translation error:", e)
        return ""


# -----------------------------------
# Display language options
# -----------------------------------
def display_language_options():
    print("\nAvailable translation languages:")
    print("1. Hindi (hi)")
    print("2. Tamil (ta)")
    print("3. Telugu (te)")
    print("4. Bengali (bn)")
    print("5. Marathi (mr)")
    print("6. Gujarati (gu)")
    print("7. Malayalam (ml)")
    print("8. Punjabi (pa)")
    print("9. Yoruba (yo)")
    print("10. Hausa (ha)")
    print("11. Igbo (ig)")

    choice = input("Select target language number: ").strip()

    languages = {
        "1": "hi",
        "2": "ta",
        "3": "te",
        "4": "bn",
        "5": "mr",
        "6": "gu",
        "7": "ml",
        "8": "pa",
        "9": "yo",
        "10": "ha",
        "11": "ig"
    }

    return languages.get(choice, "es")  # default Spanish


# -----------------------------------
# Main program
# -----------------------------------
def main():
    print("=== Speech Translator ===")

    while True:
        # Select target language
        target_language = display_language_options()

        # Capture speech
        original_text = speech_to_text()

        if original_text:
            # Translate
            translated_text = translate_text(
                original_text,
                target_language
            )

            if translated_text:
                # Speak translation
                speak(translated_text)
                print("Translation spoken successfully!")

        # Continue?
        again = input("\nTranslate another sentence? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


# -----------------------------------
# Run program
# -----------------------------------
if __name__ == "__main__":
    main()