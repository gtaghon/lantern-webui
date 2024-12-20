# LANTERN webUI
Streamlit UI for training NIST [LANTERN](https://github.com/usnistgov/lantern) machine learning model.

## Prerequisites
1. Install LANTERN on the deployment system (see [Instructions](https://github.com/usnistgov/lantern?tab=readme-ov-file#installation-guide) from LANTERN repo).
2. Install streamlit on the deployment system:
   ```
   pip install streamlit
   ```

## Deployment and Usage
```
streamlit run lantern_server.py
```
will launch a webUI running at http://localhost:8501.

### Usage
1. Drag and drop CSV phenotype data with column labels = phenotypes, variance ('phenotype_var').
2. Select desired phenotypes for training from dropdown menu. Matching variance columns are automatically selected.
3. Choose the number of epochs to train for (standard 1000 in publications).
4. Specify an output directory on the server (ie. output/myrun1)
5. Click 'Start Training'
6. When complete, download model files (`model.pt`, `loss.pt`) using UI or copy from output directory.
