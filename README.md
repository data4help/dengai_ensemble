# dengai2
Second try on the DengAI Challenge from DrivenData.org. The link to the challenge is provided here:
https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/. This challenge has the aim to predict the
cases of dengue fever in to cities. This disease is transmitted through mosquitoes, therefore predicting the number of
dengue fever is highly correlated with predicting the number of mosquitoes.

# Overview of the repo
## models
This repository basically consists out of multiple sub-repositories. In the `model` folder we find differet subfolders,
which each contain a prediction attempt. As of now the following models have been implemented:

- SARIMAX

And the following are planned and yet to come:

- Classification, Regression and ETS Combination model
- Theta Model

## data
The data folder contains the untouched raw data straight from the website. All preprocessed data for a specific model
is then saved within each model folder.

