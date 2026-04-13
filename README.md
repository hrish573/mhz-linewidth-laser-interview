# Vienna Interview

# mHz-Linewidth Laser Interview Code

This repository contains the code used to reproduce and explore results for the bad-cavity superradiant laser model discussed in the assigned paper.

## Repository structure

- `src/` — implementation of the model, solver, plotting, and utilities
- `tests/` — validation tests
- `requirements.txt` — Python dependencies

## Setup

```bash
git clone https://github.com/hrish573/mhz-linewidth-laser-interview.git
cd mhz-linewidth-laser-interview
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Run
python -m src.main
