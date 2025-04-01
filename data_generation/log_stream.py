class LogStream:
    def __init__(self, name: str):
        self.name = name

    def write(self, message: str):
        print(f"[{self.name}]: {message}")