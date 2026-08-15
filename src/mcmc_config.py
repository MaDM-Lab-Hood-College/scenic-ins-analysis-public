from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class McmcConfig:
    cores: int
    chains: int
    tune: int
    draws: int
