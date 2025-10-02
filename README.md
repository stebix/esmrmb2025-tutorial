# Programmatic Sequence Optimization

This demonstration code and notebook repository displays programmatic magnetic resonance fingerprinting optimization with respect to various
configurable loss functions and information theoretic measures like the  Cramer Rao Lower Bound (CRLB).

## Installation

### Quick Start with uv (Recommended)

This project uses `uv` for fast and reliable Python package management. If you're new to `uv`, you'll love how it simplifies dependency installation!

**First-time uv users:**
```bash
# Install uv (it's blazing fast!)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS/Linux with homebrew:
brew install uv
```

**Install the project:**
```bash
# Clone the repository
git clone <repository-url>
cd esmrmb-notebook

# Create virtual environment and install all dependencies
uv sync

# Activate the environment and start Jupyter
uv run jupyter notebook notebooks/esmrmb-demo.ipynb
```


### Alternative: Traditional pip installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook notebooks/esmrmb-demo.ipynb
```

### Developers

- Tom Griesler, University of Michigan, tomgr@umich.edu
- Dr. Martin Blaimer, Fraunhofer IIS EZRT, martin.blaimer@iis.fraunhofer.de
- Jannik Stebani, EP5, University of Würzburg, jannik.stebani@uni-wuerzburg.de


### Key publications associated with this optimization approach and context

- Griesler T, Gram M, Stebani J, Albertova P, Dawood P, Seiberlich N, ... & Blaimer M Towards Sequence Optimization for Multi-Compartment Magnetic Resonance Fingerprinting. Proceedings of the ISMRM 2024, Singapure; Abstract 354 (2024). (https://archive.ismrm.org/2024/3547.html)

- Stebani J, Angelov I, Dawood P, Griesler T, Albertova P, Kampf T, ... & Gram M. High-resolution quantitative imaging of the inner ear using 3D Magnetic Resonance Fingerprinting. Proceedings of the ISMRM 2025, Honolulu, Hawaii, USA (2025) (https://archive.ismrm.org/2025/0926_DZjs0wQ5F.html)

### Our projects are based on the following publications:

Quantitative MRI optimization based on the Cramer-Rao Lower Bound:
- Lee PK, Watkins LE, Anderson TI, Buonincontri G, Hargreaves BA. Flexible and efficient optimization of quantitative sequences using automatic differentiation of Bloch simulations. Magn Reson Med. 2019 Oct;82(4):1438-1451. doi: 10.1002/mrm.27832.

MRF optimizated based on tissue discrimiation cost function:
- Cohen O, Rosen MS. Algorithm comparison for schedule optimization in MR fingerprinting. Magn Reson Imaging. 2017 Sep;41:15-21. doi: 10.1016/j.mri.2017.02.010.

Multi-compartment MRF Optimization:
- Heesterbeek DGJ, Koolstra K, van Osch MJP, van Gijzen MB, Vos FM, Nagtegaal MA. Mitigating undersampling errors in MR fingerprinting by sequence optimization. Magn Reson Med. 2023 May;89(5):2076-2087. doi: 10.1002/mrm.29554.

Basis FA pattern
-  Cao X, Liao C, Iyer SS, et al. Optimized multi-axis spiral projection MR fingerprinting with subspace reconstruction for rapid whole-brain high-isotropic-resolution quantitative imaging. Magn Reson Med. 2022; 88: 133-150. doi:10.1002/mrm.29194