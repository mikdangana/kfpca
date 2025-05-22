from ekf import *;


class Ksurf:
    def __init__(self, nmsmt=2, dx=2, n_components=10, att_fname=None, att_col=None):
        self.kf = PCAKalmanFilter(
            nmsmt=nmsmt,
            dx=dx,
            n_components=n_components,
            normalize=True,
            att_fname=att_fname,
            att_col=att_col
        )
        self.X = []
        self.y = []

    def update(self, X, y):
        """Store action-context X and target y, and update EKF."""
        self.X.extend(X)
        self.y.extend(y)

        # Normalize and update EKF
        for i in range(1, len(self.y)):
            msmts = [self.y[i - 1], self.y[i]]
            if is_pca_or_akf():
                msmts = self.kf.pca_normalize(msmts, is_scalar=False)
                self.kf.to_H([[x, x] for x in self.X[i - 1:i + 1]], [msmts])
            self.kf.update([msmts])

    def predict(self, X):
        """Use PCA attention to generate predictions for each X row."""
        if len(self.y) < 2:
            return np.zeros(len(X)), np.ones(len(X))  # no prior data

        # Apply PCA attention
        attn_input = np.array(self.y[-self.kf.n_components:])
        values = self.kf.pca_attention([attn_input], n=self.kf.n_components)

        # Return predicted value and dummy std (EKF doesn't give std)
        mean = np.array([values[-1][-1]] * len(X))
        std = np.array([1.0] * len(X))  # placeholder
        return mean, std

    def reset(self):
        self.__init__()

