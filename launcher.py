from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import TextIO

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import psutil
except ImportError:
    psutil = None


BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "app.py"
OUTPUT_DIR = BASE_DIR / "output"
PORT = 8501
URL = f"http://localhost:{PORT}"

streamlit_process: subprocess.Popen[bytes] | None = None
stdout_log_handle: TextIO | None = None
stderr_log_handle: TextIO | None = None
is_starting = False
is_stopping = False

root: tk.Tk
status_var: tk.StringVar
log_text: ScrolledText
start_button: ttk.Button | None = None
browser_button: ttk.Button | None = None
stop_button: ttk.Button | None = None


def find_python_executable() -> str:
    candidates = [
        BASE_DIR / ".venv" / "Scripts" / "python.exe",
        BASE_DIR / "venv" / "Scripts" / "python.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if sys.executable and Path(sys.executable).exists():
        return sys.executable

    return "python"


def is_port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return True
    except OSError:
        return False


def is_streamlit_running() -> bool:
    return streamlit_process is not None and streamlit_process.poll() is None


def run_on_ui(callback, *args) -> None:
    try:
        root.after(0, callback, *args)
    except tk.TclError:
        pass


def append_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line)
    log_text.configure(state="normal")
    log_text.insert("end", line + "\n")
    log_text.see("end")
    log_text.configure(state="disabled")


def log_message(message: str) -> None:
    run_on_ui(append_log, message)


def set_status(message: str) -> None:
    def update() -> None:
        status_var.set(f"Status: {message}")
        refresh_buttons()

    run_on_ui(update)


def notify_info(message: str) -> None:
    run_on_ui(messagebox.showinfo, "Foto 10x15 Fácil", message)


def refresh_buttons() -> None:
    if start_button is None or browser_button is None or stop_button is None:
        return

    running = is_streamlit_running()

    if is_starting or is_stopping or running:
        start_button.state(["disabled"])
    else:
        start_button.state(["!disabled"])

    if is_stopping or not running:
        stop_button.state(["disabled"])
    else:
        stop_button.state(["!disabled"])

    browser_button.state(["!disabled"])


def close_log_handles() -> None:
    global stdout_log_handle, stderr_log_handle

    for handle in (stdout_log_handle, stderr_log_handle):
        if handle is None:
            continue
        try:
            handle.close()
        except OSError:
            pass

    stdout_log_handle = None
    stderr_log_handle = None


def start_program() -> None:
    global is_starting

    if is_starting:
        log_message("O programa já está iniciando.")
        return

    if is_streamlit_running() or is_port_open():
        set_status("Programa em execução")
        log_message("O programa já está em execução.")
        return

    is_starting = True
    refresh_buttons()
    threading.Thread(target=start_streamlit_worker, daemon=True).start()


def start_streamlit_worker() -> None:
    global is_starting, streamlit_process, stdout_log_handle, stderr_log_handle

    log_message("Iniciando programa...")
    set_status("iniciando")

    if not APP_FILE.exists():
        log_message("Erro ao iniciar: app.py não foi encontrado.")
        set_status("Erro ao iniciar")
        is_starting = False
        run_on_ui(refresh_buttons)
        return

    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stdout_log_handle = (OUTPUT_DIR / "launcher_streamlit_stdout.log").open(
            "a", encoding="utf-8"
        )
        stderr_log_handle = (OUTPUT_DIR / "launcher_streamlit_stderr.log").open(
            "a", encoding="utf-8"
        )

        python_executable = find_python_executable()
        command = [
            python_executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(PORT),
        ]

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        streamlit_process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=stdout_log_handle,
            stderr=stderr_log_handle,
            creationflags=creationflags,
        )
    except Exception as exc:
        close_log_handles()
        streamlit_process = None
        log_message("Erro ao iniciar. Verifique se as dependências estão instaladas.")
        print(f"Erro ao iniciar Streamlit: {exc}")
        set_status("Erro ao iniciar")
        is_starting = False
        run_on_ui(refresh_buttons)
        return

    time.sleep(3)

    if is_streamlit_running():
        set_status("Programa em execução")
        log_message("Programa iniciado.")
        open_browser()
    else:
        close_log_handles()
        log_message("Erro ao iniciar. Verifique se as dependências estão instaladas.")
        set_status("Erro ao iniciar")

    is_starting = False
    run_on_ui(refresh_buttons)


def open_browser() -> None:
    log_message("Abrindo navegador...")
    threading.Thread(target=webbrowser.open, args=(URL,), daemon=True).start()


def stop_program() -> None:
    global is_stopping

    if not is_streamlit_running():
        set_status("parado")
        log_message("Nenhum programa em execução.")
        notify_info("Nenhum programa em execução.")
        return

    is_stopping = True
    refresh_buttons()
    log_message("Encerrando programa...")
    threading.Thread(target=stop_streamlit_worker, daemon=True).start()


def stop_streamlit_worker() -> None:
    global is_stopping, streamlit_process

    process = streamlit_process
    if process is not None:
        terminate_process_tree(process)

    streamlit_process = None
    close_log_handles()
    set_status("Programa encerrado")
    log_message("Programa encerrado.")
    notify_info("Programa encerrado com sucesso.")

    is_stopping = False
    run_on_ui(refresh_buttons)


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    if psutil is None:
        terminate_without_psutil(process)
        return

    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return

    children = []
    try:
        children = parent.children(recursive=True)
    except (psutil.Error, OSError):
        pass

    for child in children:
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

    try:
        _, alive_children = psutil.wait_procs(children, timeout=5)
    except (psutil.Error, OSError):
        alive_children = children

    for child in alive_children:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

    try:
        parent.terminate()
        parent.wait(timeout=5)
    except psutil.TimeoutExpired:
        try:
            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        pass


def terminate_without_psutil(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except OSError:
        pass


def check_process_status() -> None:
    global streamlit_process

    if streamlit_process is not None and streamlit_process.poll() is not None:
        streamlit_process = None
        close_log_handles()
        if not is_starting and not is_stopping:
            set_status("Programa encerrado")
            log_message("Programa encerrado.")

    root.after(1000, check_process_status)


def on_close() -> None:
    global streamlit_process

    if is_streamlit_running():
        should_close = messagebox.askyesno(
            "Encerrar programa",
            "O programa ainda está em execução. Deseja encerrar?",
            parent=root,
        )
        if not should_close:
            return

        terminate_process_tree(streamlit_process)
        streamlit_process = None
        close_log_handles()

    root.destroy()


def center_window(window: tk.Tk, width: int, height: int) -> None:
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")


def build_ui() -> tk.Tk:
    global root, status_var, log_text, start_button, browser_button, stop_button

    root = tk.Tk()
    root.title("Foto 10x15 Fácil")
    root.resizable(False, False)
    center_window(root, 420, 300)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
    style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
    style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
    style.configure("Action.TButton", font=("Segoe UI", 10), padding=8)

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Foto 10x15 Fácil", style="Title.TLabel").pack(anchor="w")
    ttk.Label(frame, text="Controle do programa", style="Subtitle.TLabel").pack(
        anchor="w", pady=(0, 10)
    )

    status_var = tk.StringVar(value="Status: parado")
    ttk.Label(frame, textvariable=status_var, style="Status.TLabel").pack(
        anchor="w", pady=(0, 10)
    )

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(0, 10))

    start_button = ttk.Button(
        buttons, text="Iniciar programa", command=start_program, style="Action.TButton"
    )
    start_button.pack(fill="x", pady=(0, 6))

    browser_button = ttk.Button(
        buttons, text="Abrir no navegador", command=open_browser, style="Action.TButton"
    )
    browser_button.pack(fill="x", pady=(0, 6))

    stop_button = ttk.Button(
        buttons, text="Encerrar programa", command=stop_program, style="Action.TButton"
    )
    stop_button.pack(fill="x")

    log_text = ScrolledText(frame, height=5, wrap="word", font=("Segoe UI", 9))
    log_text.pack(fill="both", expand=True)
    log_text.insert("end", "Pronto para iniciar.\n")
    log_text.configure(state="disabled")

    refresh_buttons()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(1000, check_process_status)
    return root


if __name__ == "__main__":
    build_ui().mainloop()
