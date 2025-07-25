#!/usr/bin/env python3
"""
Project initialization script for new ML projects.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def create_new_project(project_name):
    """
    Create a new ML project with standard structure.
    
    Args:
        project_name (str): Name of the new project
    """
    print(f"🚀 Creating new ML project: {project_name}")
    
    # Create project directory
    project_dir = Path(project_name)
    if project_dir.exists():
        print(f"❌ Project directory '{project_name}' already exists!")
        return False
    
    project_dir.mkdir()
    
    # Create subdirectories
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "notebooks",
        "src",
        "models",
        "reports/figures",
        "experiments",
        "tests",
        "configs"
    ]
    
    for dir_path in directories:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}/")
    
    # Create initial files
    files_to_create = {
        "README.md": create_readme_content(project_name),
        "requirements.txt": create_requirements_content(),
        ".gitignore": create_gitignore_content(),
        "src/__init__.py": "",
        "src/config.py": create_config_content(),
        "notebooks/01_data_exploration.ipynb": "{}",
        ".env.template": create_env_template()
    }
    
    for file_path, content in files_to_create.items():
        full_path = project_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Created: {file_path}")
    
    print(f"\n🎉 Project '{project_name}' created successfully!")
    print(f"\nNext steps:")
    print(f"1. cd {project_name}")
    print(f"2. python -m venv venv")
    print(f"3. venv\\Scripts\\activate  # Windows")
    print(f"4. pip install -r requirements.txt")
    print(f"5. jupyter lab")
    
    return True

def create_readme_content(project_name):
    """Create README.md content."""
    return f"""# {project_name}

Machine Learning project created on {datetime.now().strftime('%Y-%m-%d')}

## Project Structure

```
{project_name}/
├── data/
│   ├── raw/          # Original data
│   ├── processed/    # Cleaned data
│   └── external/     # External datasets
├── notebooks/        # Jupyter notebooks
├── src/             # Source code
├── models/          # Trained models
├── reports/         # Generated reports
├── experiments/     # ML experiments
├── tests/          # Unit tests
└── configs/        # Configuration files
```

## Getting Started

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter:
   ```bash
   jupyter lab
   ```

## Project Goals

- [ ] Data collection and cleaning
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Model development
- [ ] Model evaluation
- [ ] Model deployment

## Contributors

- [Your Name]

## License

This project is licensed under the MIT License.
"""

def create_requirements_content():
    """Create requirements.txt content."""
    return """# Core Data Science
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
requests>=2.31.0

# Development
pytest>=7.4.0
black>=23.0.0
"""

def create_gitignore_content():
    """Create .gitignore content."""
    return """# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Virtual Environment
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints/

# Data files
data/raw/*
data/external/*
!data/raw/.gitkeep
!data/external/.gitkeep

# Models
models/*.pkl
models/*.joblib
models/*.h5

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/
"""

def create_config_content():
    """Create config.py content."""
    return '''"""Project configuration."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Visualization
FIGURE_SIZE = (10, 6)
DPI = 300
'''

def create_env_template():
    """Create .env.template content."""
    return """# Environment Variables Template
# Copy to .env and fill in your values

# API Keys
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_token_here

# Database
DATABASE_URL=your_database_url

# Project Settings
PROJECT_NAME=ML_Project
ENVIRONMENT=development
DEBUG=True
"""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python new_project.py <project_name>")
        sys.exit(1)
    
    project_name = sys.argv[1]
    create_new_project(project_name)
