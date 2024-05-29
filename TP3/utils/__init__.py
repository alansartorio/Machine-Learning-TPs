from typing import Generic, TypeVar, List, Optional
from random import seed, random


def random_range(start: float, end: float) -> float:
    if end < start:
        raise ValueError('End cannot be greater than start')
    return start + random() * (end-start)


def sign(n: float) -> int:
    return 1 if n >= 0 else -1


T = TypeVar('T')

class Collection(Generic[T]):
    def __init__(self) -> None:
        self.data: List[T] = []
        self.size = 0
        super().__init__()

    def add(self, item: T) -> None:
        self.data.append(item)
        self.size += 1 

    def pop(self) -> Optional[T]:
        if self.is_empty():
            return None
        self.size -= 1
        return self.data.pop()

    def is_empty(self) -> bool:
        return self.size == 0
    
    def __iter__(self):
        return self.data
    
    def __next__(self):
        return next(self.data)
    
    def __len__(self) -> int:
        return self.size