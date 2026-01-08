WIP code for agentifying petscagent-bench using A2A and MCP standards.

## Project Structure

```
├── data
│   └── problems_test.jsonl
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── green_agent # Assessment manager agent
│   ├── launcher.py # Evaluation coordinator
│   ├── util
│   └── white_agent # Target agent being tested
├── tools
│   └── petsc_mcp_servers
```

## Cloning the Repository

This repository includes a Git submodule.

To clone the repository along with its submodule, use:
```bash
git clone --recursive <repo-url>
```

If you have already cloned the repository without the option `--recursive`, initialize and update the submodule by running the following command from the repository root:

```bash
git submodule update --init
```

## Installation

Install the required dependencies using `uv`:

```bash
uv sync
```

## Usage

To test the code locally, make sure PETSc is installed.
Then create a file `.env` in the root folder with the following variables 

```
GEMINI_API_KEY="<your_key>"
PETSC_DIR="<your_petsc_dir>"
PETSC_ARCH="<your_petsc_arch>"
```


```bash
# Launch complete evaluation
uv run main.py launch
```
