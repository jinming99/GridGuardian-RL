# GridGuardian-RL — Hands-on tutorials and Gymnasium environments for safe and robust RL in power and energy systems

GridGuardian-RL provides step-by-step tutorials, reference environments, and a fair-comparison evaluation protocol to study safety-constrained control and robust reinforcement learning in power and energy systems, starting with EV charging.


## Why this repo

- Safety-constrained control and robust RL emphasis
- Fair-comparison framework and evaluation protocol
- Educational, step-by-step tutorials (primary entry point)


## Quickstart

1) Create and activate a Python 3.10–3.12 environment.
2) Install tutorial dependencies:
   ```bash
   pip install -r tutorials/requirements-tutorials.txt
   ```
3) Open the tutorials (primary entry point):
   - Start in `tutorials/` or jump directly to the hub: [tutorials/README.md](tutorials/README.md)
   - Launch JupyterLab/Notebook and open a flagship tutorial such as:
     - Standard RL + robustness: [tutorials/03_ev_standard_rl_robustnes.ipynb](tutorials/03_ev_standard_rl_robustnes.ipynb)
     - Safe RL with CMDPs: [tutorials/04_ev_safe_rl_cmdp.ipynb](tutorials/04_ev_safe_rl_cmdp.ipynb)


## Tutorials index

- Hub (recommended): [tutorials/README.md](tutorials/README.md)
- Start here (2–4 suggested):
  - [01_ev_env_safety_robustness.ipynb](tutorials/01_ev_env_safety_robustness.ipynb)
  - [02_ev_baselines_and_trajectories.ipynb](tutorials/02_ev_baselines_and_trajectories.ipynb)
  - [03_ev_standard_rl_robustnes.ipynb](tutorials/03_ev_standard_rl_robustnes.ipynb)
  - [04_ev_safe_rl_cmdp.ipynb](tutorials/04_ev_safe_rl_cmdp.ipynb)


## Environments and data

- SustainGym — Base EV charging environment, adapted with robustness wrappers: [https://chrisyeh96.github.io/sustaingym/](https://chrisyeh96.github.io/sustaingym/) ([Yeh et al., NeurIPS 2023](https://openreview.net/forum?id=vZ9tA3o3hr))
- ACN-Data/ACN-Sim — Real charging session data from Caltech: [https://github.com/zach401/acnportal](https://github.com/zach401/acnportal) ([Lee et al., 2019](https://doi.org/10.1145/3307772.3331015))
- Robust-Gymnasium — Robustness testing framework: [https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/Robust-Gymnasium) ([Gu et al., ICLR 2025](https://openreview.net/forum?id=example))

Local environment code used in tutorials: `envs/evcharging/`


## Citation and license

- Citation file: [CITATION.cff](CITATION.cff)
- License: [LICENSE](LICENSE)


## Roadmap

- Expand robust evaluation harness (e.g., Robust-Gymnasium sweeps and adversarial tests)
- Extend fair-comparison protocol and standardized seeds across algorithms
- Add more energy-system tasks and multi-agent tutorials
- Improve environment packaging and documentation
- Advanced safe RL algorithms and optimization-as-a-layer experiments
