# roblox-up-and-coming-forecast-01

Monte Carlo latent-strength forecast of the Roblox Up-and-Coming top 10 — a make.fun bounty entry.

## What it does

`main.py` (Python 3.13, standard library only) reads `/input/history.ndjson` and writes `/output/forecast.json`, forecasting the next Roblox Up-and-Coming top 10 by simulating latent strengths with a Monte Carlo method.

## Usage

```sh
python3 main.py
```

Input: `/input/history.ndjson`  
Output: `/output/forecast.json`

## License

MIT — see [LICENSE](LICENSE).
