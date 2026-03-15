# visualizations/hello_world.py
from tkinter import messagebox

VIZ_NAME = "Testing Visualization"

def run(xlsx_path: str) -> None:
    messagebox.showinfo(
        "Hello World",
        f"Visualization module executed successfully.\n\nSelected file:\n{xlsx_path}"
    )