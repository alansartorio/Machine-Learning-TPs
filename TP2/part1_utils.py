from abc import ABC, abstractmethod
from typing import List, Dict, Generic, TypeVar, Iterable

T = TypeVar("T")


class Range(ABC, Generic[T]):
    @abstractmethod
    def inside_range(self, value: T) -> bool:
        pass


class IntRange(Range[int]):
    def __init__(
        self, min: int, max: int, min_inclusive=True, max_inclusive=False
    ) -> None:
        if min > max:
            raise Exception("Min value cannot be bigger than max value for range")
        self.min = min
        self.max = max
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

    def inside_range(self, value: int) -> bool:
        if value < self.min or value > self.max:
            return False
        if value == self.min:
            return self.min_inclusive
        if value == self.max:
            return self.max_inclusive
        return True

    def __inside_range_min(self, value: int) -> bool:
        return value < self.min if self.min_inclusive else value <= self.min

    def __str__(self) -> str:
        return f"{'[' if self.min_inclusive else '('}{self.min}, {self.max}{']' if self.max_inclusive else ')'}"


class VariableBalancer(Generic[T]):
    def __init__(self, ranges: List[Range[T]], default_range: Range[T]) -> None:
        self.ranges = list(enumerate(ranges, 1))
        self.default_range = (0, default_range)
        self.ranges.append(self.default_range)

    def balanced_value_to_str(self, value: int) -> str:
        if not isinstance(value, int):
            return None
        for v, r in self.ranges:
            if value == v:
                return str(r)
        return str(self.default_range[1])

    def get_balanced_values(self) -> Iterable[int]:
        return [v for v, r in self.ranges]

    def get_mappings(self) -> List[Dict[int, str]]:
        return [{v: str(r)} for v, r in self.ranges]

    def balance(self, value: T) -> int:
        for i, r in self.ranges:
            if r.inside_range(value):
                return i
        return self.default_range[0]


if __name__ == "__main__":
    balancer = VariableBalancer(
        [
            IntRange(0, 18),
            IntRange(18, 30),
            IntRange(30, 50),
        ],
        IntRange(50, float("inf")),
    )

    import json

    print(json.dumps(balancer.get_mappings(), indent=4))

    print(balancer.balance(10))
    print(balancer.balance(23))
    print(balancer.balance(60))
    print(balancer.balance(45))
