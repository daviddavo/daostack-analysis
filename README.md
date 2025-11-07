# The Rise and Fall of DAOstack: Lessons for Decentralized Autonomous Organizations

This repository contains the code and data analysis for the research paper "_The Rise and Fall of DAOstack: Lessons for Decentralized Autonomous Organizations_" by David Davó, Javier Arroyo, Samer Hassan, and Silvia Semenzin, published in [PeerJ Computer Science](http://dx.doi.org/10.7717/peerj-cs.3320).

To cite it, please see the [Citation](#citation) section.

## About

This study presents the first postmortem analysis of DAOstack, a pioneering DAO platform that operated from 2017 to 2023. Using a mixed-methods approach combining quantitative blockchain data analysis and qualitative interviews, we examine the rise and eventual abandonment of this first-generation DAO platform to extract lessons for future decentralized governance systems.

## Paper Abstract

> Despite the hype and scandals around blockchain, there are valuable applications beyond Finance, such as Decentralized Autonomous Organizations (DAOs). DAOs are self-governed online communities where users vote and manage budgets transparently. In under a decade, DAOs have evolved from theory to managing billions of dollars. Blockchain enthusiasts launched DAO platforms like our case study, "DAOstack", promising large-scale collaboration and quickly securing millions in funding. Today, we can critically evaluate to what extent the platform followed up on its promises. In this work, we analyze DAOstack using a mixed-methods approach combining quantitative and qualitative data. In particular, we quantitatively examined its 92 organizations in terms of size, lifespan, activity, power concentration, and the effectiveness of its governance model.  We also interviewed in-depth 6 DAOstack core users to delve deep into their experiences using the platform.  Our analysis shows that DAOstack mainly hosted small, short-lived DAOs, with some exceptions. Its governance model was functional, but the economic incentives underpinning it were ineffective. The analysis of the interviews reveals interesting aspects such as the power imbalances due to token ownership and reputation, and that the voting system, though innovative, was affected by issues of cost and complexity.
We conclude by discussing the challenges these platforms face and advocating for a multidisciplinary experimental approach for future DAO designers.

## Repository Structure

### `/datawarehouse/`
Contains the processed blockchain data from DAOstack:

- `daos.arr` - Information about deployed DAOs
- `proposals.arr` - Proposal data and voting outcomes
- `votes.arr` - Individual vote records
- `stakes.arr` - Staking/prediction market data
- `reputationHolders.arr`, `reputationMints.arr`, `reputationBurns.arr` - Reputation system data

### `/notebooks/`
Jupyter notebooks containing the analysis:

#### Main Analysis
- `0 Prepare Data.ipynb` - Data preparation and cleaning
- `index.ipynb` - Main analysis summary and overview
- `proposals.ipynb` - Proposal analysis and voting patterns
- `voting.ipynb` - Voting behavior analysis
- `DAOs.ipynb` - Individual DAO characterization
- `activity.ipynb` - Platform activity over time
- `equality.ipynb` - Power concentration analysis (Gini coefficient, Nakamoto coefficient)
- `holders.ipynb` - Reputation holder analysis
- `stakers.ipynb` - Staking behavior analysis
- `boosting_predictor.ipynb` - Holographic Consensus effectiveness analysis
- `queue.ipynb` - Proposal queue analysis

#### Specific Studies
- `11J.ipynb` - Analysis of the July 11, 2020 events and its effect on the network.
- `dxDAO.ipynb` - In-depth analysis of dxDAO
- `downstaking.ipynb` - Downstaking behavior analysis
- `hc-params.ipynb` - Holographic Consensus parameters study
- `unregistered.ipynb` - Analysis of unregistered DAOs

#### Utilities
- `utils/` - Python utility modules:
  - `dw.py` - Data warehouse access functions
  - `functions.py` - Common analysis functions
  - `plot.py` - Plotting utilities
  - `tables.py` - Table generation utilities
- `aux/` - Auxiliary notebooks to report bugs or understand inner workings of libraries
- `common.ipy` - Common IPython functions and imports. It's automatically run at the start of each notebook.

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start with `notebooks/0 Prepare Data.ipynb` to populate the cache of auxiliary data. Needs to be run each time the Python version changes.
4. Explore individual analysis notebooks based on your interests

The output of the notebooks is also provided, so it is not required to run them. Note that some annotations might be in Spanish.

## Data

The analysis covers DAOstack's entire lifespan from April 2019 to May 2023, including:
- 92 DAO deployments (25 on Ethereum mainnet, 67 on xDAI)
- Over 9,000 user addresses
- Thousands of proposals and votes
- Complete staking and reputation data

Data was collected from [DAO-Analyzer](https://dao-analyzer.science).

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{Dav2025,
  title = {The rise and fall of DAOstack: lessons for decentralized autonomous organizations},
  volume = {11},
  ISSN = {2376-5992},
  url = {http://dx.doi.org/10.7717/peerj-cs.3320},
  DOI = {10.7717/peerj-cs.3320},
  journal = {PeerJ Computer Science},
  publisher = {PeerJ},
  author = {Davó,  David and Arroyo,  Javier and Hassan,  Samer and Semenzin,  Silvia},
  year = {2025},
  month = nov,
  pages = {e3320}
}
```

You can also cite this repository directly:

```bibtex
@software{davo2024daostack_analysis,
  author = {Davó, David},
  title = {daviddavo/daostack-analysis},
  url = {https://github.com/daviddavo/daostack-analysis},
  doi = {10.5281/zenodo.16420191},
  year = {2024}
}
```

## License

The article site is licensed under a Creative Commons Attribution 4.0 International License. The code is licensed under GPLv3.

## Acknowledgments

<div align="center">
<img src="./logos/logo-ministerio.png"
     alt="Logo Ministerio de Ciencia e Innovación. Gobierno de España"
     style="max-height: 3em"
><img src="./logos/logo-grasia.png"
     alt="Logo GRASIA UCM"
     style="max-height: 3em"
><img src="./logos/logo-ucm.png"
     alt="Logo Universidad Complutense de Madrid"
     style="max-height: 3em"
>
</div>

This research was supported by the Spanish Ministry of Science and Innovation under Grant PID2021-127956OB-I00 (Project DAO Applications).