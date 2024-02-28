from optbinning import OptimalBinning

class CustomDiscretizer:
    def __init__(self, variable, dtype='numerical', max_n_bins=10, min_bin_size=0.05):
        self.variable = variable
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.optbinner = OptimalBinning(name=self.variable, 
                                        dtype=dtype, 
                                        solver="cp", 
                                        max_n_bins=self.max_n_bins, 
                                        min_bin_size=self.min_bin_size)

    def fit(self, X, y):
        self.optbinner.fit(X[self.variable], y)  # Pass only the specified variable to OptBinning
        self.binning_table = self.optbinner.binning_table
        return self

    def transform(self, X):
        return self.optbinner.transform(X[self.variable])

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
