import scipy.io
import pandas as pd

mat = scipy.io.loadmat(
    "/Users/jamesthornton/ers/EnsembleRejectionSampler/data/sp500returns.mat"
)
print(mat["obs"].shape)

sp500 = pd.read_csv(
    "/Users/jamesthornton/ers/EnsembleRejectionSampler/data/sp500.txt", sep="\t"
)
print(sp500.shape)
print(sp500.head())

pd.DataFrame({"returns": mat["obs"][0]}).to_csv(
    "/Users/jamesthornton/ers/EnsembleRejectionSampler/data/sp500returns.csv",
    index=False,
)
