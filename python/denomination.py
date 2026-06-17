from tkinter import *
from tkinter import messagebox

# -----------------------------
# Main Window
# -----------------------------
root = Tk()
root.title("Denomination Counter")
root.geometry("650x450")
root.configure(bg="light blue")

# -----------------------------
# Header Frame
# -----------------------------
header = Frame(root, bg="navy", height=120)
header.pack(fill=X)

title_label = Label(
    header,
    text="DENOMINATION COUNTER",
    font=("Arial", 24, "bold"),
    bg="navy",
    fg="white"
)
title_label.pack(pady=15)

subtitle_label = Label(
    header,
    text="Count Currency Notes Quickly and Easily",
    font=("Arial", 12),
    bg="navy",
    fg="white"
)
subtitle_label.pack()

# -----------------------------
# Welcome Message
# -----------------------------
welcome_label = Label(
    root,
    text="Welcome to the Denomination Counter Application",
    font=("Arial", 14, "bold"),
    bg="light blue"
)
welcome_label.pack(pady=30)

info_label = Label(
    root,
    text="Click the button below to calculate denomination counts.",
    font=("Arial", 11),
    bg="light blue"
)
info_label.pack()

# -----------------------------
# Denomination Window
# -----------------------------
def topwin():

    top = Toplevel(root)
    top.title("Denominations Calculator")
    top.geometry("600x380")
    top.configure(bg="light grey")

    Label(
        top,
        text="Enter Total Amount",
        font=("Arial", 12, "bold"),
        bg="light grey"
    ).place(x=220, y=40)

    entry = Entry(top, width=25)
    entry.place(x=200, y=75)

    Label(
        top,
        text="Number of Notes for Each Denomination",
        font=("Arial", 12, "bold"),
        bg="light grey"
    ).place(x=130, y=150)

    Label(top, text="2000", bg="light grey").place(x=180, y=200)
    Label(top, text="500", bg="light grey").place(x=180, y=235)
    Label(top, text="100", bg="light grey").place(x=180, y=270)

    t1 = Entry(top)
    t2 = Entry(top)
    t3 = Entry(top)

    t1.place(x=270, y=200)
    t2.place(x=270, y=235)
    t3.place(x=270, y=270)

    def calculator():
        try:
            amount = int(entry.get())

            note2000 = amount // 2000
            amount %= 2000

            note500 = amount // 500
            amount %= 500

            note100 = amount // 100

            t1.delete(0, END)
            t2.delete(0, END)
            t3.delete(0, END)

            t1.insert(0, str(note2000))
            t2.insert(0, str(note500))
            t3.insert(0, str(note100))

        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid amount."
            )

    Button(
        top,
        text="Calculate",
        command=calculator,
        bg="green",
        fg="white",
        font=("Arial", 10, "bold")
    ).place(x=250, y=110)

# -----------------------------
# Start Button Function
# -----------------------------
def msg():
    messagebox.showinfo(
        "Denomination Counter",
        "Click OK to continue."
    )
    topwin()

# -----------------------------
# Main Button
# -----------------------------
Button(
    root,
    text="LET'S GET STARTED",
    command=msg,
    bg="brown",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=20,
    pady=10
).pack(pady=30)

# -----------------------------
# Footer
# -----------------------------
footer = Label(
    root,
    text="Python Tkinter Project",
    bg="light blue",
    font=("Arial", 10, "italic")
)
footer.pack(side=BOTTOM, pady=10)

# -----------------------------
# Run Program
# -----------------------------
root.mainloop()