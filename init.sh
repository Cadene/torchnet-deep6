touch README.md
mkdir data
touch data/.gitkeep
mkdir data/external
touch data/external/.gitkeep
mkdir data/interim
touch data/interim/.gitkeep
mkdir data/processed
touch data/processed/.gitkeep
mkdir data/raw
touch data/raw/.gitkeep
mkdir docs
touch docs/.gitkeep
mkdir models
touch models/.gitkeep
mkdir models/raw
touch models/raw/.gitkeep
mkdir models/processed
touch models/processed/.gitkeep
mkdir notebooks
touch notebooks/.gitkeep
mkdir reports
touch reports/.gitkeep
mkdir reports/figures
touch reports/figures/.gitkeep
touch requirements.txt
mkdir src
touch src/__init__.py
mkdir src/data
touch src/data/.gitkeep
mkdir src/features
touch src/features/.gitkeep
mkdir src/models
touch src/models/.gitkeep
mkdir src/visualization
touch src/visualization/.gitkeep
echo "# MacOs indexing files" > .gitignore
echo ".DS_Store" >> .gitignore
echo "._.DS_Store" >> .gitignore
echo "# Exclude data" >> .gitignore
echo "data/raw/*" >> .gitignore
echo "data/processed/*" >> .gitignore
echo "data/features/*" >> .gitignore
echo "data/interim/*" >> .gitignore
echo "# Exclude models" >> .gitignore
echo "models/raw/*" >> .gitignore
echo "models/processed/*" >> .gitignore
echo "!.gitkeep" >> .gitignore
