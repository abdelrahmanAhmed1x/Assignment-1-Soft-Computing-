"""Matrix order is depot first, then customers in their declared order."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Customer:
    id: int
    demand: float


@dataclass(frozen=True)
class Vehicle:
    id: int
    capacity: float


@dataclass(frozen=True)
class Route:
    vehicle_id: int
    customers: tuple[int, ...]
    load: float
    distance: float


@dataclass(frozen=True)
class CVRPInstance:
    depot: int
    customers: tuple[Customer, ...]
    vehicles: tuple[Vehicle, ...]
    distance_matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        ids = [self.depot] + [c.id for c in self.customers]
        vehicle_ids = [v.id for v in self.vehicles]
        if any(type(x) is not int for x in ids + vehicle_ids):
            raise ValueError("Location and vehicle IDs must be integers")
        if len(set(ids)) != len(ids) or len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("Location and vehicle IDs must be unique")
        if not self.customers or not self.vehicles:
            raise ValueError("At least one customer and vehicle required")
        if any(not math.isfinite(v.capacity) or v.capacity <= 0 for v in self.vehicles):
            raise ValueError("Capacities must be positive and finite")
        if any(not math.isfinite(c.demand) or c.demand < 0 for c in self.customers):
            raise ValueError("Demands must be nonnegative and finite")
        matrix = self.distance_matrix
        if len(matrix) != len(ids) or any(len(row) != len(ids) for row in matrix):
            raise ValueError("Distance matrix dimensions must match locations")
        if any(not math.isfinite(x) or x < 0 for row in matrix for x in row):
            raise ValueError("Distances must be finite and nonnegative")
        if max(c.demand for c in self.customers) > max(v.capacity for v in self.vehicles):
            raise ValueError("A customer demand exceeds every vehicle capacity")
        if sum(c.demand for c in self.customers) > sum(v.capacity for v in self.vehicles):
            raise ValueError("Total demand exceeds total fleet capacity")
