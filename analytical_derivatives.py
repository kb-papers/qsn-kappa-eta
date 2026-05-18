import numpy as np
from sklearn.pipeline import Pipeline


def analytical_derivative_eta(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    """Compute dL/dη analytically via the chain rule on the EtaRegressor weights.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline containing kappa_encoder and eta_regressor steps.
    X : ndarray, shape (n_test, n_features)
        Test features.
    y : ndarray, shape (n_test,)
        Test targets.

    Returns
    -------
    float
        Analytical gradient dL/dη.
    """
    encoder = pipeline.named_steps["kappa_encoder"]
    regressor = pipeline.named_steps["eta_regressor"]

    X_transformed = encoder.transform(X)
    y_pred = regressor.predict(X_transformed)
    error = y - y_pred

    X_train_transformed = regressor.X_train_
    y_train = regressor.y_train_
    eta = regressor.eta

    d_y_hat_d_eta = np.empty(X_transformed.shape[0])
    for i in range(X_transformed.shape[0]):
        distances = np.sqrt(np.sum((X_transformed[i] - X_train_transformed) ** 2, axis=1))
        weights = 1 / (1 + distances) ** eta
        log_dist = np.log(1 + distances)

        numerator = np.sum(weights * (y_pred[i] - y_train) * log_dist)
        denominator = np.sum(weights)
        d_y_hat_d_eta[i] = numerator / denominator if denominator != 0 else 0

    return float(-2 * np.mean(error * d_y_hat_d_eta))


def analytical_derivative_kappa_complete(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    p: int,
    X_train: np.ndarray,
) -> float:
    """Compute dL/dκ_p analytically via the full chain rule.

    Accounts for kappa's effect on both test and training encodings.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline containing kappa_encoder and eta_regressor steps.
    X : ndarray, shape (n_test, n_features)
        Test features.
    y : ndarray, shape (n_test,)
        Test targets.
    p : int
        Feature index for which to compute the kappa derivative.
    X_train : ndarray, shape (n_train, n_features)
        Training features (raw).

    Returns
    -------
    float
        Analytical gradient dL/dκ_p.
    """
    encoder = pipeline.named_steps["kappa_encoder"]
    regressor = pipeline.named_steps["eta_regressor"]

    X_transformed = encoder.transform(X)
    y_pred = regressor.predict(X_transformed)
    error = y - y_pred

    X_train_transformed = regressor.X_train_
    y_train_reg = regressor.y_train_
    eta = regressor.eta

    X_train_enc = encoder.train_col_values[p]
    y_train_enc = encoder.train_target_values
    kappa_p = encoder.kappa[p]

    total_derivative = 0.0

    for i in range(len(X)):
        # Part 1: Effect through test-data encoding  d(x̃_test)/d(kappa)
        x_i = X[i, p]
        distances_kappa = np.abs(x_i - X_train_enc)
        weights_kappa = 1 / (1 + distances_kappa) ** kappa_p
        log_dist_kappa = np.log(1 + distances_kappa)
        dw_dkappa = -weights_kappa * log_dist_kappa

        num = (
            np.sum(y_train_enc * dw_dkappa) * np.sum(weights_kappa)
            - np.sum(y_train_enc * weights_kappa) * np.sum(dw_dkappa)
        )
        den = np.sum(weights_kappa) ** 2
        d_xtilde_test_d_kappa = num / den if den != 0 else 0

        # d(ŷ_i)/d(x̃_test_i)
        tilde_x_i = X_transformed[i, 0]
        distances_test = np.abs(tilde_x_i - X_train_transformed[:, 0])
        weights_eta = 1 / (1 + distances_test) ** eta
        d_dist_d_x = np.sign(tilde_x_i - X_train_transformed[:, 0])
        dw_ddist = -eta / (1 + distances_test) ** (eta + 1)
        dw_dx = dw_ddist * d_dist_d_x

        num = (
            np.sum(y_train_reg * dw_dx) * np.sum(weights_eta)
            - np.sum(y_train_reg * weights_eta) * np.sum(dw_dx)
        )
        den = np.sum(weights_eta) ** 2
        d_yhat_d_xtilde_test = num / den if den != 0 else 0

        # Part 2: Effect through training-data encoding  d(x̃_train)/d(kappa)
        d_yhat_d_kappa_from_train = 0.0
        sum_weights = np.sum(weights_eta)
        sum_y_weights = np.sum(y_train_reg * weights_eta)

        for j in range(len(X_train)):
            x_train_j = X_train[j, p]
            distances_kappa_train = np.abs(x_train_j - X_train_enc)
            weights_kappa_train = 1 / (1 + distances_kappa_train) ** kappa_p
            log_dist_kappa_train = np.log(1 + distances_kappa_train)
            dw_dkappa_train = -weights_kappa_train * log_dist_kappa_train

            num_train = (
                np.sum(y_train_enc * dw_dkappa_train) * np.sum(weights_kappa_train)
                - np.sum(y_train_enc * weights_kappa_train) * np.sum(dw_dkappa_train)
            )
            den_train = np.sum(weights_kappa_train) ** 2
            d_xtilde_train_j_d_kappa = num_train / den_train if den_train != 0 else 0

            tilde_x_train_j = X_train_transformed[j, 0]
            dist_ij = abs(tilde_x_i - tilde_x_train_j)

            d_dist_ij_d_xtrain_j = -np.sign(tilde_x_i - tilde_x_train_j)
            d_weight_j_d_xtrain_j = (
                -eta / (1 + dist_ij) ** (eta + 1) * d_dist_ij_d_xtrain_j
            )

            d_yhat_d_xtrain_j = (
                y_train_reg[j] * d_weight_j_d_xtrain_j * sum_weights
                - sum_y_weights * d_weight_j_d_xtrain_j
            ) / sum_weights ** 2

            d_yhat_d_kappa_from_train += d_yhat_d_xtrain_j * d_xtilde_train_j_d_kappa

        d_yhat_d_kappa_total = (
            d_yhat_d_xtilde_test * d_xtilde_test_d_kappa + d_yhat_d_kappa_from_train
        )
        total_derivative += error[i] * d_yhat_d_kappa_total

    return float(-2 * total_derivative / len(X))