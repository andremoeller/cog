from pydantic import BaseModel


class Output(BaseModel):
    number: int


class Trainer:
    def train() -> Output:
        return 42

    def cancel() -> None:
        pass
