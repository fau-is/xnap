# XNAP: Making LSTM-based Next Activity Predictions Explainable by Using LRP

## General
We integrated an adapted version of the LRP approach proposed by Arras et al. (2017) -- "Explaining recurrent neural
network predictions in sentiment analysis" -- into a predictive business process monitoring technique for predicting next activities.

## Paper
If you use our code or fragments, please cite our paper:

```
@proceedings{weinzierl2020xnap,
    title={XNAP: Making LSTM-based Next Activity Predictions Explainable by Using LRP},
    author={Sven Weinzierl and Sandra Zilker and Jens Brunk and Kate Revoredo and Martin Matzner and Jörg Becker},
    booktitle={Proceedings of the 4th International Workshop on Artificial Intelligence for Business Process Management (AI4BPM2020)},
    year={2020}
}
```

You can access the paper [here](https://www.researchgate.net/publication/342918341_XNAP_Making_LSTM-based_Next_Activity_Predictions_Explainable_by_Using_LRP).

## Setup of XNAP?
   1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) 
   2. After setting up miniconda you can make use of the `conda` command in your command line (e.g. CMD or Bash)
   3. To quickly install the `xnap` package, run `pip install -e .` inside the root directory.
   4. To install required packages run `pip install -r requirements.txt` inside the root directory.
   6. Train and test the Bi-LSTM models for the next activity prediction by executing `runner.py` (note: you have to set in config.py the parameter "explain==False")
   7. Create explanations through LRP by executing `runner.py` (note: you have to set in config.py the parameter "explain==True")