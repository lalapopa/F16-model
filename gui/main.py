import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
from functools import partial
import math
import sys

from F16model.find_trim_position import run


def menu():
    global root
    global V0
    global H0
    global text_box

    root = tk.Tk()

    window_setup()
    init_value_V0 = BoxWidget("V0:")
    V0 = init_value_V0.take_number()

    init_value_H0 = BoxWidget("H0:")
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
    text_box = tk.Text(root, height=15, width=52)
    text_box.grid()

    #    checkbutton_png.button.grid()
    #    checkbutton_pgf.button.grid()
    root.mainloop()


def window_setup():
    root.title("Trim F16")
    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = int(screen_width / 3.2)
    window_height = int(screen_height / 1.8)

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)


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
    play_button.grid(sticky=tk.N, pady=10)
    return play_button


class CheckbuttonWidget:
    def __init__(self, text, text_above=None, exe_function=None):
        self.space = ttk.Frame(root)
        self.space.grid(ipady=2, sticky="n")
        if text_above:
            above_box_text = ttk.Label(self.space, text=str(text_above))
            above_box_text.grid()
        self.checkmark_state = tk.IntVar()
        self.button = ttk.Checkbutton(
            self.space,
            text=str(text),
            variable=self.checkmark_state,
            command=exe_function,
            onvalue=1,
            offvalue=0,
            width=5,
        )


class BoxWidget:
    def __init__(self, text):
        self.space = ttk.Frame(root)
        self.above_box_text = ttk.Label(self.space, text=str(text))

    def take_number(self, float_number=False):
        value = tk.StringVar()
        if float_number:
            vcmd = (
                self.space.register(partial(self._only_num_valid, float_valid=True)),
                "%S",
            )
        else:
            vcmd = (self.space.register(self._only_num_valid), "%S")
        box_entry = ttk.Entry(
            self.space, validate="key", validatecommand=vcmd, textvariable=value
        )
        self.space.grid(sticky="w", padx=72)
        self.above_box_text.grid(sticky="w")
        box_entry.grid(sticky="w")
        return value

    def _only_num_valid(self, S, float_valid=False):
        if float_valid:
            valid_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]
        else:
            valid_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        if S in valid_symbols:
            return True
        self.space.bell()
        return False


def browser_box():
    space = ttk.Frame(root)
    above_text = ttk.Label(space, text="Путь для сохранения:")
    above_text.grid(sticky="w")

    info = above_text.grid_info()
    row = info["row"]
    column = info["column"]

    box_entry = ttk.Entry(space, textvariable=folder_path, width=40)
    browse_button = ttk.Button(space, text="Browse", command=open_folder_window)

    space.grid()
    box_entry.grid(row=row + 1, column=column)
    browse_button.grid(row=row + 1, column=column + 1)


def open_folder_window():
    file_name = filedialog.askdirectory()
    folder_path.set(file_name)


def press_button():
    V0_float = Converter.float_number(V0)
    if not V0_float:
        return
    H0_float = Converter.float_number(H0)
    if not H0_float:
        return
    u0, alpha, theta = run(V0_float, H0_float)
    text_box.delete(1.0, tk.END)
    text_box.insert(
        1.0, f"{'='*10}\nH0 = {H0_float:.1f} m, V0 = {V0_float:.2f} m/s\n{'='*10}\n"
    )
    text_box.insert(
        tk.END,
        f"stab = {math.degrees(u0.stab):.4f} degree;\nthrottle = {u0.throttle:.4f} %;\nalpha = {math.degrees(alpha):.4f} degree;\n",
    )


class Converter:
    def float_number(box):
        try:
            return float(box.get())
        except:
            Notification.invalid_number()
            return False

    def step_size_box(box):
        try:
            ss = float(step_size_input.get().replace(",", "."))
        except:
            Notification.step_size_value_error()

        if 0 < ss <= 2:
            return ss
        else:
            Notification.step_size_not_in_range_error()
            return False


class Notification:
    def invalid_number():
        messagebox.showerror("Error", "Введите значение V0 и H0")

    def path_error():
        messagebox.showerror("Error", "Неверный путь сохранения!")

    def step_size_value_error():
        messagebox.showerror(
            "Error",
            "Неверное значение шага расчета!\nЧисло должно быть десятичным.\nПример: 0.1, 1.5.",
        )

    def step_size_not_in_range_error():
        messagebox.showerror("Error", "Диапазон значений шага (0,2]")

    def dont_save_plot_warning():
        state = messagebox.askquestion(
            "Warning",
            "Вы не выбрали сохранение графиков.\nЕсли продолжить графики не будут сохраняться.",
            icon="warning",
        )
        return state

    def pgf_warning():
        pgf_status = checkbutton_pgf.checkmark_state.get()
        if pgf_status:
            user_answer = Notification.run_time_warning()
            if user_answer == "no":
                checkbutton_pgf.checkmark_state.set(0)

    def run_time_warning():
        state = messagebox.askquestion(
            "Warning",
            "Файлы в pgf для LaTeX сохраняются очень долго :( \nХочешь сохранить?",
            icon="warning",
        )
        return state


if __name__ == "__main__":
    menu()
