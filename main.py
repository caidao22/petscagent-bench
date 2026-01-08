"""CLI entry point."""

import typer
import asyncio

from src.green_agent.server import start_green_agent
from src.purple_agent.petsc_agent import start_purple_agent
from src.launcher import launch_evaluation

app = typer.Typer(help="Agentified petscagent-bench - PETSc coding agent assessment framework")


@app.command()
def green():
    """Start the green agent (assessment manager)."""
    start_green_agent()


@app.command()
def purple():
    """Start the purple agent (target being tested)."""
    start_purple_agent()


@app.command()
def launch():
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation())


if __name__ == "__main__":
    app()
