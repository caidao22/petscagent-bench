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
