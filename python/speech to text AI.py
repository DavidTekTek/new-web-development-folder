import customtkinter as ctk
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import threading
import json
import os
import tempfile
import platform

# -------- SETTINGS --------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

recognizer = sr.Recognizer()

languages = {
    "Yoruba": "yo",
    "Hausa": "ha",
    "Igbo": "ig"
}

DATA_FILE = "chat_history.json"

# -------- DATA --------

def load_history():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

chat_history = load_history()

# -------- AUDIO --------

def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    system = platform.system()
    if system == "Windows":
        os.startfile(temp_file.name)
    elif system == "Darwin":
        os.system(f"afplay {temp_file.name}")
    else:
        os.system(f"xdg-open {temp_file.name}")

# -------- LOGIC --------

def process_audio():
    try:
        update_status("🎤 Listening...")

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        update_status("🧠 Recognizing...")
        text = recognizer.recognize_google(audio)

        selected_lang = lang_menu.get()
        target_lang = languages[selected_lang]

        update_status("🌍 Translating...")
        translated_text = GoogleTranslator(
            source='auto',
            target=target_lang
        ).translate(text)

        entry = {
            "input": text,
            "output": translated_text
        }

        chat_history.append(entry)
        save_history(chat_history)

        update_chat(text, translated_text)
        speak(translated_text, target_lang)

        update_status("✅ Done")

    except Exception as e:
        update_status(f"❌ Error: {str(e)}")

# -------- UI --------

def update_status(text):
    status_label.configure(text=text)
    app.update_idletasks()

def update_chat(user, bot):
    chat_box.configure(state="normal")
    chat_box.insert("end", f"You: {user}\n")
    chat_box.insert("end", f"App: {bot}\n\n")
    chat_box.configure(state="disabled")
    chat_box.see("end")

def load_chat_ui():
    for item in chat_history:
        update_chat(item["input"], item["output"])

def start_listening():
    threading.Thread(target=process_audio, daemon=True).start()

# -------- APP --------

app = ctk.CTk()
app.title("🌍 AI Translator (Compatible Version)")
app.geometry("800x650")

title = ctk.CTkLabel(app, text="🌍 AI Speech Translator", font=("Arial", 26, "bold"))
title.pack(pady=15)

top = ctk.CTkFrame(app)
top.pack(pady=10)

lang_menu = ctk.CTkOptionMenu(top, values=list(languages.keys()))
lang_menu.pack(side="left", padx=10)

listen_btn = ctk.CTkButton(
    top,
    text="🎤 Start Listening",
    command=start_listening,
    corner_radius=20,
    height=45
)
listen_btn.pack(side="left", padx=10)

chat_box = ctk.CTkTextbox(app, height=400, corner_radius=15)
chat_box.pack(padx=20, pady=10, fill="both", expand=True)
chat_box.configure(state="disabled")

status_label = ctk.CTkLabel(app, text="Ready", text_color="lightgreen")
status_label.pack(pady=10)

load_chat_ui()

app.mainloop()