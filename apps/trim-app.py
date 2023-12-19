import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from functools import partial
import math
import threading
from screeninfo import get_monitors

from F16model.utils.find_trim_position import run


class TrimApp:
    def __init__(self, parent):
        self.root = parent
        self._window_setup()
        self.V0 = BoxWidget(self.root, "V0 [m/s] = ", 0).take_number()
        self.H0 = BoxWidget(root, "H0 [m] = ", 1).take_number()
        self.run_button = self.play_button()

        self.text_box = tk.Text(root, height=10, relief="flat")
        self.text_box.configure(state="disabled")
        self.text_box.grid(row=4, columnspan=2, padx=10)

    def play_button(self):
        play_button = ttk.Button(
            self.root,
            text="Запустить",
            command=lambda: threading.Thread(target=self.press_button).start(),
        )
        play_button.grid(row=3, columnspan=2, sticky=tk.N, pady=5, padx=5)
        return play_button

    def press_button(self):
        V0_float = Converter.float_number(self.V0)
        if not V0_float:
            return
        H0_float = Converter.float_number(self.H0)
        if not H0_float:
            return

        self.disable_run_button = True
        self.handle_button(self.run_button)

        self._write_to_textbox(self.text_box, f"Running ...")
        u0, alpha, _ = run(V0_float, H0_float)
        out_text = (
            f"{'='*10}\nH0 = {H0_float:.1f} m, V0 = {V0_float:.2f} m/s\n{'='*10}\n"
        )
        out_text += f"stab = {math.degrees(u0.stab):.4f} degree;\nthrottle = {u0.throttle:.4f} %;\nalpha = {math.degrees(alpha):.4f} degree;\n"
        self._write_to_textbox(self.text_box, out_text)

        self.disable_run_button = False
        self.handle_button(self.run_button)

    def handle_button(self, button):
        if self.disable_run_button:
            button["state"] = tk.DISABLED
        else:
            button["state"] = tk.NORMAL

    def _window_setup(self):
        self.root.title("Trim F16")

        screen_width, screen_height = get_monitor_size()

        window_width = int(screen_width / 4.2)
        window_height = int(screen_height / 2.8)

        # set the position of the window to the center of the screen
        self.root.geometry(f"{290}x{window_height}")
        self.root.resizable(1, 1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

    def _write_to_textbox(self, tb, text):
        tb.configure(state="normal")
        tb.delete(1.0, tk.END)
        tb.insert(1.0, text)
        tb.configure(state="disabled")


def get_monitor_size():
    for m in get_monitors():
        if m.is_primary:
            return m.width, m.height
    return (1920, 1080)


class BoxWidget:
    def __init__(self, root, text, order_number):
        self.root = root
        self.order = order_number
        self.label = ttk.Label(self.root, text=str(text))
        self.label.grid(row=self.order, column=0, padx=5, pady=10, sticky=tk.E)

    def take_number(self, float_number=False):
        value = tk.StringVar()
        if float_number:
            vcmd = (
                self.root.register(partial(self._only_num_valid, float_valid=True)),
                "%S",
            )
        else:
            vcmd = (self.root.register(self._only_num_valid), "%S")
        box_entry = ttk.Entry(
            self.root,
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
    root = tk.Tk()
    TrimApp(root)
    root.mainloop()
