import customtkinter as ctk
import speech_recognition as sr
import pyttsx3
from datetime import datetime
import threading

# ---------- INIT ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

recognizer = sr.Recognizer()

# ---------- FUNCTIONS ----------

def speak(text):
    output_box.configure(state="normal")
    output_box.insert("end", "Assistant: " + text + "\n\n")
    output_box.configure(state="disabled")
    output_box.see("end")

    engine.say(text)
    engine.runAndWait()


def listen():
    status_label.configure(text="Listening...")
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()

            output_box.configure(state="normal")
            output_box.insert("end", "You: " + command + "\n")
            output_box.configure(state="disabled")

            respond(command)

        except sr.WaitTimeoutError:
            status_label.configure(text="Listening timeout")
        except sr.UnknownValueError:
            speak("I could not understand you.")
        except sr.RequestError:
            speak("Network error occurred.")

    status_label.configure(text="Ready")


def respond(command):
    if "hello" in command:
        speak("Hi there! How can I help you?")
    
    elif "your name" in command:
        speak("I am your Python voice assistant.")
    
    elif "time" in command:
        now = datetime.now().strftime("%H:%M")
        speak(f"The time is {now}")
    
    elif "exit" in command or "stop" in command:
        speak("Goodbye!")
        app.quit()
    
    else:
        speak("I am not sure how to help with that.")


def start_listening():
    threading.Thread(target=listen, daemon=True).start()


# ---------- UI ----------

app = ctk.CTk()
app.title("Voice Assistant")
app.geometry("600x500")

title = ctk.CTkLabel(app, text="Voice Assistant", font=("Arial", 24, "bold"))
title.pack(pady=15)

output_box = ctk.CTkTextbox(app, height=250, corner_radius=15)
output_box.pack(padx=20, pady=10, fill="both", expand=True)
output_box.configure(state="disabled")

listen_btn = ctk.CTkButton(
    app,
    text="Start Listening",
    command=start_listening,
    height=50,
    corner_radius=20,
    font=("Arial", 16, "bold")
)
listen_btn.pack(pady=15)

status_label = ctk.CTkLabel(app, text="Ready", text_color="lightgreen")
status_label.pack(pady=5)

# Welcome message
speak("Voice assistant ready. Click the button and speak.")

# ---------- RUN ----------
app.mainloop()