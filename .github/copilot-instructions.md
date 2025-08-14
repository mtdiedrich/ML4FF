# ML4FF - Machine Learning for Fantasy Football

ML4FF is a Python-based machine learning project for fantasy football analysis, specifically focused on rookie NFL player projections. The main codebase includes rookie projection pipelines, Jupyter notebooks for experimentation, and machine learning models.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- Bootstrap Python environment and dependencies:
  - Verify Python 3.12+ is available: `python3 --version`
  - Install core ML dependencies: `pip3 install pandas numpy scikit-learn jupyter` -- takes 90 seconds. NEVER CANCEL. Set timeout to 5+ minutes.
  - Install testing framework: `pip3 install pytest` -- takes 10 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- Test installation:
  - `python3 -c "import pandas as pd, numpy as np, sklearn; print('ML libraries available')"` -- verifies all packages work
  - `python3 -m pytest --version` -- verifies testing framework

### Project Structure
Based on branch analysis, the full project contains:
- `src/` - Main Python source code including `rookie_projection_pipeline.py`
- `notebooks/` - Jupyter notebooks for experimentation and usage examples
- Root level contains standard Python project files (.gitignore, README.md)

### Development Workflow
- Create virtual environment (recommended but not required):
  - `python3 -m venv venv`
  - `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip3 install pandas numpy scikit-learn jupyter pytest`
- Run notebooks: `jupyter notebook` or `jupyter lab` -- starts server on port 8888. NEVER CANCEL once started.
- Execute notebooks: `jupyter nbconvert --to notebook --execute notebook.ipynb --output executed.ipynb`

### Testing
- Run all tests: `python3 -m pytest -v` -- takes less than 1 second. Set timeout to 1+ minutes.
- Run specific test file: `python3 -m pytest tests/test_filename.py -v`
- Test basic ML functionality: 
  ```python
  python3 -c "
  import pandas as pd
  import numpy as np
  df = pd.DataFrame({'a': [1,2,3]})
  print('Basic test passed:', len(df) == 3)
  "
  ```
- Test comprehensive ML workflow:
  ```python
  python3 -c "
  import pandas as pd, numpy as np, sklearn.linear_model as lm
  data = pd.DataFrame({'x': [1,2,3,4,5], 'y': [2,4,6,8,10]})
  model = lm.LinearRegression().fit(data[['x']], data['y'])
  print('ML workflow test passed:', model.score(data[['x']], data['y']) > 0.9)
  "
  ```
  ```

### Validation
- ALWAYS test new ML code with sample data before committing.
- ALWAYS run through at least one complete data processing scenario after making changes.
- Test notebook functionality by starting Jupyter server and running cells interactively.
- For ML pipeline changes, verify the rookie projection pipeline can load and process sample data.
- Programmatic notebook execution may have syntax issues; prefer interactive testing via Jupyter server.

## Common Tasks

### Working with Notebooks
- Start Jupyter server: `jupyter notebook` -- launches on http://localhost:8888. NEVER CANCEL.
- Convert notebook to script: `jupyter nbconvert --to script notebook.ipynb`
- Execute notebook programmatically: `jupyter nbconvert --to notebook --execute notebook.ipynb`
- **Note**: Programmatic notebook execution may have formatting issues. Prefer interactive execution via Jupyter server for testing.

### Working with the ML Pipeline
- The main pipeline is in `src/rookie_projection_pipeline.py` (available in rookie-projection branch)
- Usage pattern based on commit history:
  ```python
  from rookie_projection_pipeline import RookieProjectionPipeline
  pipeline = RookieProjectionPipeline(season=2024)
  results = pipeline.run()
  # Save results: pipeline.run(save_csv=True)
  ```

### Data and Dependencies
- Pandas for data manipulation - standard DataFrame operations
- NumPy for numerical computing - array operations and math
- Scikit-learn for machine learning - models and preprocessing
- Jupyter for interactive development and experimentation

## Repository State Information
- **Current branch (copilot/fix-9)**: Minimal repository with just .gitignore and README.md
- **Main development branch (rookie-projection)**: Contains full ML pipeline codebase with src/ and notebooks/ directories
- **Project focus**: NFL fantasy football rookie player performance projections using machine learning

## Timing Expectations
- **Package installation**: 90 seconds for core ML packages (pandas, numpy, scikit-learn, jupyter). NEVER CANCEL - set 5+ minute timeout.
- **Testing**: Under 1 second for basic tests. Set 1+ minute timeout for safety.
- **Notebook execution**: Varies by complexity, typically 5-30 seconds. Set 2+ minute timeout.
- **Jupyter server startup**: 2-5 seconds. NEVER CANCEL once running.

## Important Notes
- This repository uses the Python ecosystem standard structure
- Focus is on NFL fantasy football data and rookie projections
- Main development appears to happen in feature branches (rookie-projection)
- Always verify ML library imports work before developing new features
- Use Jupyter notebooks for experimentation and data exploration
- The codebase follows standard Python ML project patterns with src/ for modules and notebooks/ for analysis