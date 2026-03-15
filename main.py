# main.py
#
# small tkinter launcher for the survey visualizations. scans the ./visualizations/
# folder on startup and populates the dropdown with anything that has VIZ_NAME and run().
# adding a new visualization just means dropping a new .py file in that folder
# and hitting refresh.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import importlib
import pkgutil
import traceback


def load_visualizations():
    # walk the visualizations package and collect anything that looks like a viz module.
    # modules starting with _ get skipped (thats the __init__ etc.)
    viz_options = {}

    pkg_name = "visualizations"
    try:
        package = importlib.import_module(pkg_name)
    except Exception as e:
        messagebox.showerror(
            "Visualization load error",
            f"Could not import the visualizations package.\n"
            f"Make sure the visualizations folder exists and has an __init__.py.\n\n{e}",
        )
        return viz_options

    for modinfo in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        mod_name = modinfo.name

        if mod_name.split(".")[-1].startswith("_"):
            continue

        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # dont crash the whole launcher if one viz has a broken import
            print(f"couldnt load {mod_name}, skipping:\n{traceback.format_exc()}")
            continue

        viz_name = getattr(mod, "VIZ_NAME", None)
        run_fn   = getattr(mod, "run", None)

        if isinstance(viz_name, str) and callable(run_fn):
            viz_options[viz_name] = run_fn

    return viz_options


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Survey Visualization")
        self.geometry("680x240")
        self.minsize(680, 240)

        self.selected_file = tk.StringVar(value="")
        self.selected_viz  = tk.StringVar(value="")

        # name -> run() function, rebuilt on refresh
        self.viz_options = load_visualizations()

        self.build_ui()
        self.bind_state_updates()
        self.update_run_btn()

    def build_ui(self):
        pad = {"padx": 12, "pady": 8}

        ttk.Label(
            self,
            text="Run a visualization on an Excel dataset",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", **pad) # pyright: ignore[reportArgumentType]

        ttk.Label(self, text="Visualization:").grid(row=1, column=0, sticky="w", **pad) # pyright: ignore[reportArgumentType]

        viz_names = sorted(self.viz_options.keys())
        self.viz_combo = ttk.Combobox(
            self,
            textvariable=self.selected_viz,
            values=viz_names,
            state="readonly",
            width=52,
        )
        self.viz_combo.grid(row=1, column=1, columnspan=2, sticky="we", **pad) # pyright: ignore[reportArgumentType]

        ttk.Label(self, text="Excel file (.xlsx):").grid(row=2, column=0, sticky="w", **pad) # pyright: ignore[reportArgumentType]

        self.file_entry = ttk.Entry(self, textvariable=self.selected_file, width=52)
        self.file_entry.grid(row=2, column=1, sticky="we", **pad) # pyright: ignore[reportArgumentType]

        ttk.Button(self, text="Browse...", command=self.browse_file).grid(row=2, column=2, sticky="e", **pad) # pyright: ignore[reportArgumentType]

        self.run_btn = ttk.Button(self, text="Run visualization", command=self.run_selected)
        self.run_btn.grid(row=3, column=1, sticky="e", **pad) # pyright: ignore[reportArgumentType]

        ttk.Button(self, text="Refresh list", command=self.refresh_vizs).grid(row=3, column=0, sticky="w", **pad) # pyright: ignore[reportArgumentType]
        ttk.Button(self, text="Quit", command=self.destroy).grid(row=3, column=2, sticky="e", **pad) # pyright: ignore[reportArgumentType]

        self.grid_columnconfigure(1, weight=1)

        status_text = (
            "No visualizations found. Add .py files in ./visualizations with VIZ_NAME and run()."
            if not self.viz_options
            else "Select a visualization and an .xlsx file, then click Run."
        )
        self.status = ttk.Label(self, text=status_text)
        self.status.grid(row=4, column=0, columnspan=3, sticky="w", **pad)  # pyright: ignore[reportArgumentType]

    def bind_state_updates(self):
        self.viz_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_run_btn())
        self.selected_file.trace_add("write", lambda *_: self.update_run_btn())

    def refresh_vizs(self):
        self.viz_options = load_visualizations()
        viz_names = sorted(self.viz_options.keys())

        self.viz_combo["values"] = viz_names
        self.selected_viz.set("")  # clear so stale selection doesnt stay highlighted
        self.update_run_btn()

        self.status.config(text=(
            "No visualizations found. Add .py files in ./visualizations."
            if not viz_names
            else "Visualization list refreshed."
        ))

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if path:
            self.selected_file.set(path)

    def update_run_btn(self):
        file_ok = self.valid_xlsx(self.selected_file.get())
        viz_ok  = self.selected_viz.get() in self.viz_options
        self.run_btn.config(state=("normal" if (file_ok and viz_ok) else "disabled"))

    def valid_xlsx(self, path_str):
        if not path_str:
            return False
        p = Path(path_str)
        return p.exists() and p.suffix.lower() == ".xlsx"

    def run_selected(self):
        viz_name  = self.selected_viz.get()
        xlsx_path = self.selected_file.get()

        if viz_name not in self.viz_options:
            messagebox.showerror("Missing selection", "Please select a visualization.")
            return
        if not self.valid_xlsx(xlsx_path):
            messagebox.showerror("Invalid file", "Please select a valid .xlsx file.")
            return

        self.status.config(text=f"Running: {viz_name}")
        self.update_idletasks()  # flush the status label update before the viz blocks the thread

        try:
            self.viz_options[viz_name](xlsx_path)
            self.status.config(text=f"Done: {viz_name}")
        except Exception as e:
            self.status.config(text="Error while running visualization.")
            messagebox.showerror(
                "Run failed",
                f"Visualization failed:\n\n{e}\n\nDetails:\n{traceback.format_exc()}",
            )


if __name__ == "__main__":
    try:
        # windows dpi fix - makes text not look blurry on high-res displays.
        # silently ignored on non-windows
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = App()
    app.mainloop()