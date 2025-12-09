from sklearn.base import BaseEstimator, TransformerMixin

## Custom transformer for Feature Engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = None):
        self.threshold = threshold

    def fit(self, X, y = None):
        if self.threshold is None:
            self.threshold = X["amount"].mean() + 3 * X["amount"].std()
        return self

    def transform(self, X):
        X = X.copy()

        # Balance changes for sender and receiver
        X["balanceDiffOrig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
        X["balanceDiffDest"] = X["newbalanceDest"] - X["oldbalanceDest"]

        # Large transaction flag
        X["isLargeTransaction"] = (X["amount"] > self.threshold).astype("int32")

        # Zero balance flag for sender
        X["zero_balance_flag"] = ((X["oldbalanceOrg"] > 0) & (X["newbalanceOrig"] == 0) &
                           (X["type"].isin(["TRANSFER", "CASH_OUT"]))).astype("int32")

        # Relative transaction amounts for sender and receiver
        X["sender_amount_ratio"] = X["amount"] / (X["oldbalanceOrg"] + 1).astype("float32")
        X["receiver_amount_ratio"] = X["amount"] / (X["oldbalanceDest"] + 1).astype("float32")

        keep_cols = ["type", "amount", "balanceDiffOrig", "balanceDiffDest",
                     "isLargeTransaction", "zero_balance_flag",
                     "sender_amount_ratio", "receiver_amount_ratio"]

        return X[keep_cols]