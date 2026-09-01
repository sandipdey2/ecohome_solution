"""EcoHome Energy Advisor package."""

__version__ = "1.1.0"
__all__ = ["Agent", "EnergyAdvisor", "build_advisor_graph"]


def __getattr__(name: str):
    if name in {"Agent", "EnergyAdvisor", "build_advisor_graph"}:
        from ecohome.agent import Agent, EnergyAdvisor, build_advisor_graph

        return {
            "Agent": Agent,
            "EnergyAdvisor": EnergyAdvisor,
            "build_advisor_graph": build_advisor_graph,
        }[name]
    raise AttributeError(name)
