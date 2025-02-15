"""
Streamlit WebUI for training LANTERN models
Dec 2024
"""
import streamlit as st
from lantern.dataset import Dataset
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model.likelihood import MultitaskGaussianLikelihood
from lantern.loss import ELBO_GP
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


# Compute device detection
if torch.backends.mps.is_available():
    device = "mps"
    st.write("Training on MPS GPU.")
elif torch.cuda.is_available():
    device = "cuda"
    st.write("Training on CUDA GPU.")
else:
    device = "cpu"

def train_lantern(
    loader, optimizer, loss, model, epochs, output_dir
):
    """General purpose lantern training optimization loop"""

    record = {}
    try:
        with st.spinner("Training..."):
            pbar = st.progress(0)
            lossdata = pd.DataFrame(columns=['epoch', 'loss'])
            # Updates graph per epoch
            placeholder = st.empty()

            for e in range(epochs):

                # logging of loss values
                tloss = 0
                vloss = 0

                # store total loss terms
                terms = {}

                # go through minibatches
                for btch in loader:
                    optimizer.zero_grad()
                    yhat = model(btch[0])
                    lss = loss(yhat, *btch[1:])

                    for k, v in lss.items():
                        if k not in terms:
                            terms[k] = []
                        terms[k].append(v.item())

                    total = sum(lss.values())
                    total.backward()

                    optimizer.step()
                    tloss += total.item()

                # update record
                record[e] = {}
                for k, v in terms.items():
                    record[e][k] = np.mean(v)

                # update log
                pbar.progress((e + 1) / epochs)
                current_loss = {'epoch':e, 'loss': tloss / len(loader)}
                lossdata.loc[len(lossdata)] = current_loss
                
                with placeholder.container():
                    chart = st.line_chart(lossdata, x='epoch', y='loss')

                    if e % 10 == 0:
                        note = st.write(f"Epoch {e+1}, Loss: {tloss / len(loader)}")

            # Save training results
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            torch.save(loss.state_dict(), os.path.join(output_dir, "loss.pt"))

    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model-error.pt"))
        torch.save(loss.state_dict(), os.path.join(output_dir, "loss-error.pt"))
        raise e

def main():
    st.title("LANTERN Trainer")

    # Get data file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        column_names = [col.strip() for col in df.columns if "_var" not in col]

        # Select phenotypes
        phenotype_cols = st.multiselect("Select Phenotype Columns:", column_names)

        # Create error columns
        error_cols = [col + "_var" for col in phenotype_cols]

        # Get training parameters
        epochs = st.number_input("Number of Epochs", min_value=1, value=1000, step=100)
        batch_size = st.number_input("Batch Size", min_value=4096, value=8192, step=1024)
        lr = 0.01

        output_dir = st.text_input("Output Directory", value="output")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Start training button
        start_button = st.button("Start Training")
        if start_button:
            ds = Dataset(df, phenotypes=phenotype_cols, errors=error_cols)
            model = Model(
                VariationalBasis.fromDataset(ds, 8, meanEffectsInit=False),
                Phenotype.fromDataset(ds, 8),
                MultitaskGaussianLikelihood(len(phenotype_cols))
            )

            if device != "cpu":
                ds.to(device)
                model = model.to(device)
            else:
                print("WARNING: no accelerator found. Proceeding with CPU.")

            loss = model.basis.loss(N=len(ds),) + ELBO_GP.fromModel(
                model, len(ds),
            )
            optimizer = Adam(loss.parameters(), lr=lr)
            loader = DataLoader(ds, batch_size=batch_size)

            train_lantern(loader, optimizer, loss, model, epochs, output_dir)

if __name__ == "__main__":
    main()