import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
from functools import partial
from screeninfo import get_monitors
import math
import sys

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

    #     step_size_calc_box = BoxWidget("Шаг расчета H:")
    #     step_size_input = step_size_calc_box.take_number(float_number=True)
    #
    #     folder_path = tk.StringVar()
    #     browser_box()
    #     checkbutton_png = CheckbuttonWidget("png", "Сохранить графики в формате:")
    #     checkbutton_pgf = CheckbuttonWidget("pgf", exe_function=Notification.pgf_warning)

    # text_box.insert("end", str(button.invoke()) + "\n")
    play_button_setup()
    text_box = tk.Text(root, height=10, relief="flat")
    text_box.configure(state="disabled")
    text_box.grid(row=4, columnspan=2, padx=10)

    #    checkbutton_png.button.grid()
    #    checkbutton_pgf.button.grid()

    root.mainloop()


def window_setup():
    global window_width
    global window_height

    root.title("Trim F16")

    screen_width, screen_height = get_monitor_size()

    window_width = int(screen_width / 4.2)
    window_height = int(screen_height / 2.8)

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

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
    #     button_icon = Image.open("../assets/button_img.png")
    #     button_icon = button_icon.resize((40, 40), Image.ANTIALIAS)
    #     button_icon = ImageTk.PhotoImage(button_icon)
    play_button = ttk.Button(
        root,
        text="Запустить",
        command=press_button,
    )
    #    play_button.button_icon = button_icon
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
    u0, alpha, theta = run(V0_float, H0_float)
    text_box.configure(state="normal")
    text_box.delete(1.0, tk.END)
    text_box.insert(
        1.0, f"{'='*10}\nH0 = {H0_float:.1f} m, V0 = {V0_float:.2f} m/s\n{'='*10}\n"
    )
    text_box.insert(
        tk.END,
        f"stab = {math.degrees(u0.stab):.4f} degree;\nthrottle = {u0.throttle:.4f} %;\nalpha = {math.degrees(alpha):.4f} degree;\n",
    )
    text_box.configure(state="disabled")


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