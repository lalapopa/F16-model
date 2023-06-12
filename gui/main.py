import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from functools import partial
import math
import threading
from screeninfo import get_monitors

from F16model.find_trim_position import run


def menu():
    global root
    global V0
    global H0
    global text_box
    global text_V0
    global text_H0

    root = tk.Tk()
    window_setup()

    init_value_V0 = BoxWidget("V0 = ", 0)
    V0 = init_value_V0.take_number()

    init_value_H0 = BoxWidget("H0 = ", 1)
    H0 = init_value_H0.take_number()

    play_button_setup()
    text_box = tk.Text(root, height=10, relief="flat")
    text_box.configure(state="disabled")
    text_box.grid(row=4, columnspan=2, padx=10)

    root.mainloop()


def window_setup():
    global window_width
    global window_height

    root.title("Trim F16")

    screen_width, screen_height = get_monitor_size()

    window_width = int(screen_width / 4.2)
    window_height = int(screen_height / 2.8)

    # set the position of the window to the center of the screen
    root.geometry(f"{290}x{window_height}")
    root.resizable(1, 1)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)


def get_monitor_size():
    for m in get_monitors():
        if m.is_primary:
            return m.width, m.height
    return (1920, 1080)


def play_button_setup():
    play_button = ttk.Button(
        root,
        text="Запустить",
        command=threading.Thread(target=press_button).start,
    )
    play_button.grid(row=3, columnspan=2, sticky=tk.N, pady=5, padx=5)
    return play_button


class BoxWidget:
    def __init__(self, text, order_number):
        self.order = order_number
        self.label = ttk.Label(root, text=str(text))
        self.label.grid(row=self.order, column=0, padx=5, pady=10, sticky=tk.E)

    def take_number(self, float_number=False):
        value = tk.StringVar()
        if float_number:
            vcmd = (
                root.register(partial(self._only_num_valid, float_valid=True)),
                "%S",
            )
        else:
            vcmd = (root.register(self._only_num_valid), "%S")
        box_entry = ttk.Entry(
            root,
            validate="key",
            validatecommand=vcmd,
            textvariable=value,
        )
        box_entry.grid(row=self.order, column=1, padx=5, pady=5, sticky=tk.W)
        return value

    def _only_num_valid(self, S, float_valid=False):
        if float_valid:
            valid_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]
        else:
            valid_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        if S in valid_symbols:
            return True
        return False


def press_button():
    V0_float = Converter.float_number(V0)
    if not V0_float:
        return
    H0_float = Converter.float_number(H0)
    if not H0_float:
        return
    run_text = f"Running ..."
    write_to_textbox(text_box, run_text)
    u0, alpha, _ = run(V0_float, H0_float)
    out_text = f"{'='*10}\nH0 = {H0_float:.1f} m, V0 = {V0_float:.2f} m/s\n{'='*10}\n"
    out_text += f"stab = {math.degrees(u0.stab):.4f} degree;\nthrottle = {u0.throttle:.4f} %;\nalpha = {math.degrees(alpha):.4f} degree;\n"
    write_to_textbox(text_box, out_text)

def write_to_textbox(tb, text):
    tb.configure(state="normal")
    tb.delete(1.0, tk.END)
    tb.insert(
        1.0, text
    )
    tb.configure(state="disabled")

class Converter:
    def float_number(box):
        try:
            return float(box.get())
        except:
            Notification.invalid_number()
            return False


class Notification:
    def invalid_number():
        messagebox.showerror("Error", "Введите значение V0 и H0")


if __name__ == "__main__":
    menu()
