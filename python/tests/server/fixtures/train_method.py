from pydantic import BaseModel


class Output(BaseModel):
    number: int


def train() -> Output:
    return 42
