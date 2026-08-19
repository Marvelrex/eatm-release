from rich.console import Console

class Logger:
    _console = Console()
    _DEBUG_MODE = False
    _log_file = None

    @staticmethod
    def set_debug_mode(debug_mode):
        Logger.DEBUG_MODE = debug_mode

    @staticmethod
    def set_log_file(path: str):
        Logger._log_file = open(path, "a")

    @staticmethod
    def _write_to_file(text: str):
        if Logger._log_file:
            try:
                Logger._log_file.write(text + "\n")
                Logger._log_file.flush()
            except Exception:
                pass

    @staticmethod
    def info(message: str):
        Logger._console.print(f"[INFO] {message}", style="bold blue")
        Logger._write_to_file(f"[INFO] {message}")

    @staticmethod
    def warning(message: str):
        Logger._console.print(f"[!] {message}", style="bold yellow")
        Logger._write_to_file(f"[!] {message}")

    @staticmethod
    def error(message: str):
        Logger._console.print(f"[X] {message}", style="bold red")
        Logger._write_to_file(f"[X] {message}")

    @staticmethod
    def success(message: str):
        Logger._console.print(f"[+] {message}", style="bold green")
        Logger._write_to_file(f"[+] {message}")

    @staticmethod
    def debug(message: str):
        if Logger.DEBUG_MODE:
            Logger._console.print(f"[DEBUG] {message}", style="white")
        Logger._write_to_file(f"[DEBUG] {message}")

    @staticmethod
    def conversation_message(role: str, message: str):
        TRUNCATE_LENGTH = 420
        truncated_message = (message[:TRUNCATE_LENGTH] + '...' + '[TRUNCATED]') if len(message) > TRUNCATE_LENGTH else message
        Logger._console.print(f"[bold magenta][{role.upper()}][/bold magenta] {truncated_message}")
        Logger._console.print("---------------------", style="bold magenta")
        Logger._write_to_file(f"[{role.upper()}] {message}")
        Logger._write_to_file("---------------------")

    @staticmethod
    def new_episode():
        Logger._console.print("---------------------", style="bold yellow")
        Logger._console.print("Starting new episode", style="bold yellow")
        Logger._console.print("---------------------", style="bold yellow")
        Logger._write_to_file("=====================")
        Logger._write_to_file("Starting new episode")
        Logger._write_to_file("=====================")
