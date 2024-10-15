import warnings

import numpy as np
import pandas as pd
import sklearn.preprocessing

NONE_REPLACEMENT = -32768


def information_mutual_conditional(
    x,
    y,
    z,
    cartesian_product=False,
    base=2,
    fill_value=-1,
    estimator="ML",
    alphabet_x=None,
    alphabet_y=None,
    Alphabet_Z=None,
    keep_dims=False,
):
    x, fill_value_X = _sanitise_array_input(x, fill_value)
    y, fill_value_Y = _sanitise_array_input(y, fill_value)
    z, fill_value_Z = _sanitise_array_input(z, fill_value)
    if alphabet_x is not None:
        alphabet_x, fill_value_Alphabet_X = _sanitise_array_input(
            alphabet_x, fill_value
        )
        alphabet_x, _ = _autocreate_alphabet(alphabet_x, fill_value_Alphabet_X)
    else:
        alphabet_x, fill_value_Alphabet_X = _autocreate_alphabet(x, fill_value_X)
    if alphabet_y is not None:
        alphabet_y, fill_value_Alphabet_Y = _sanitise_array_input(
            alphabet_y, fill_value
        )
        alphabet_y, _ = _autocreate_alphabet(alphabet_y, fill_value_Alphabet_Y)
    else:
        alphabet_y, fill_value_Alphabet_Y = _autocreate_alphabet(y, fill_value_Y)
    if Alphabet_Z is not None:
        Alphabet_Z, fill_value_Alphabet_Z = _sanitise_array_input(
            Alphabet_Z, fill_value
        )
        Alphabet_Z, _ = _autocreate_alphabet(Alphabet_Z, fill_value_Alphabet_Z)
    else:
        Alphabet_Z, fill_value_Alphabet_Z = _autocreate_alphabet(z, fill_value_Z)

    if x.size == 0:
        raise ValueError("arg X contains no elements")
    if y.size == 0:
        raise ValueError("arg Y contains no elements")
    if z.size == 0:
        raise ValueError("arg Z contains no elements")
    if np.any(_isnan(x)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(y)):
        raise ValueError("arg Y contains NaN values")
    if np.any(_isnan(z)):
        raise ValueError("arg Z contains NaN values")
    if alphabet_x.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(alphabet_x)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if alphabet_y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(alphabet_y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if Alphabet_Z.size == 0:
        raise ValueError("arg Alphabet_Z contains no elements")
    if np.any(_isnan(Alphabet_Z)):
        raise ValueError("arg Alphabet_Z contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if _isnan(fill_value_Z):
        raise ValueError("fill value for arg Z is NaN")
    if x.shape[:-1] != alphabet_x.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not " "match")
    if y.shape[:-1] != alphabet_y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not " "match")
    if z.shape[:-1] != Alphabet_Z.shape[:-1]:
        raise ValueError("leading dimensions of args Z and Alphabet_Z do not " "match")
    if not cartesian_product and (x.shape != y.shape or x.shape != z.shape):
        raise ValueError("dimensions of args X, Y, Z do not match")
    if cartesian_product and (x.shape[-1] != y.shape[-1] or x.shape[-1] != z.shape[-1]):
        raise ValueError("trailing dimensions of args X, Y, Z do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers(
        (x, alphabet_x, y, alphabet_y, z, Alphabet_Z),
        (
            fill_value_X,
            fill_value_Alphabet_X,
            fill_value_Y,
            fill_value_Alphabet_Y,
            fill_value_Z,
            fill_value_Alphabet_Z,
        ),
    )
    x, alphabet_x, y, alphabet_y, z, Alphabet_Z = S

    if not cartesian_product:
        I = np.empty(x.shape[:-1])
        if I.ndim > 0:
            I[:] = np.NaN
        else:
            I = np.float64(np.NaN)
    else:
        shapeI_Z = z.shape[:-1]
        z = np.reshape(z, (-1, z.shape[-1]))
        Alphabet_Z = np.reshape(Alphabet_Z, (-1, Alphabet_Z.shape[-1]))
        I = []
        for i in range(z.shape[0]):

            def f(X, Y, Alphabet_X, Alphabet_Y):
                return information_mutual_conditional(
                    X,
                    Y,
                    z[i],
                    False,
                    base,
                    fill_value,
                    estimator,
                    Alphabet_X,
                    Alphabet_Y,
                    Alphabet_Z[i],
                )

            I.append(_cartesian_product_apply(x, y, f, alphabet_x, alphabet_y))
        shapeI_XY = I[0].shape
        if len(shapeI_Z) == 0:
            I = np.array(I)[0].reshape(shapeI_XY)
        else:
            I = np.array(I)
            I = np.rollaxis(I, 0, len(I.shape))
            I = I.reshape(np.append(shapeI_XY, shapeI_Z).astype("int"))
        return I

    # Re-shape H, X,Y,Z so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    x = np.reshape(x, (-1, x.shape[-1]))
    y = np.reshape(y, (-1, y.shape[-1]))
    z = np.reshape(z, (-1, z.shape[-1]))
    alphabet_x = np.reshape(alphabet_x, (-1, alphabet_x.shape[-1]))
    alphabet_y = np.reshape(alphabet_y, (-1, alphabet_y.shape[-1]))
    Alphabet_Z = np.reshape(Alphabet_Z, (-1, Alphabet_Z.shape[-1]))
    orig_shape_I = I.shape
    I = np.reshape(I, (-1, 1))

    for i in range(x.shape[0]):
        I_ = (
            entropy_joint(
                np.vstack((x[i], z[i])),
                base,
                fill_value,
                estimator,
                _vstack_pad((alphabet_x[i], Alphabet_Z[i]), fill_value),
            )
            + entropy_joint(
                np.vstack((y[i], z[i])),
                base,
                fill_value,
                estimator,
                _vstack_pad((alphabet_y[i], Alphabet_Z[i]), fill_value),
            )
            - entropy_joint(
                np.vstack((x[i], y[i], z[i])),
                base,
                fill_value,
                estimator,
                _vstack_pad((alphabet_x[i], alphabet_y[i], Alphabet_Z[i]), fill_value),
            )
            - entropy_joint(z[i], base, fill_value, estimator, Alphabet_Z[i])
        )
        I[i] = I_

    # Reverse re-shaping
    I = np.reshape(I, orig_shape_I)

    if keep_dims and not cartesian_product:
        I = I[..., np.newaxis]

    return I


def information_mutual(
    X,
    Y=None,
    cartesian_product=False,
    base=2,
    fill_value=-1,
    estimator="ML",
    Alphabet_X=None,
    Alphabet_Y=None,
    keep_dims=False,
):
    H_conditional = entropy_conditional(
        X, Y, cartesian_product, base, fill_value, estimator, Alphabet_X, Alphabet_Y
    )
    H = entropy(X, base, fill_value, estimator, Alphabet_X)

    H = np.reshape(
        H, np.append(H.shape, np.ones(H_conditional.ndim - H.ndim)).astype("int")
    )

    H = H - H_conditional

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def entropy_pmf(P, base=2, require_valid_pmf=True, keep_dims=False):
    P, _ = _sanitise_array_input(P)

    if P.size == 0:
        raise ValueError("arg P contains no elements")
    if np.any(_isnan(P)):
        raise ValueError("arg P contains NaN values")
    if np.any(np.logical_or(P < 0, P > 1)):
        raise ValueError("arg P contains values outside unit interval")
    if require_valid_pmf and not np.allclose(np.sum(P, axis=-1), 1):
        raise ValueError("arg P does not sum to unity across last axis")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    H = -np.sum(P * np.log2(P + np.spacing(0)), axis=-1)
    H = H / np.log2(base)

    if keep_dims:
        H = H[..., np.newaxis]

    return H


def entropy_joint(
    X, base=2, fill_value=-1, estimator="ML", Alphabet_X=None, keep_dims=False
):
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(
            Alphabet_X, fill_value
        )
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X, fill_value_X)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not " "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers(
        (X, Alphabet_X), (fill_value_X, fill_value_Alphabet_X)
    )
    X, Alphabet_X = S

    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)

    for i in range(X.shape[0]):
        X = X[:, X[i].argsort(kind="mergesort")]

    B = np.any(X[:, 1:] != X[:, :-1], axis=0)
    I = np.append(np.where(B), X.shape[1] - 1)
    L = np.diff(np.append(-1, I))

    alphabet_X = X[:, I]
    if estimator != "ML":
        n_additional_empty_bins = _determine_number_additional_empty_bins(
            L, alphabet_X, Alphabet_X, fill_value
        )
    else:
        n_additional_empty_bins = 0
    L, _ = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
    if not np.any(L):
        return np.float64(np.NaN)

    # P_0 is the probability mass assigned to each additional empty bin
    P, P_0 = _estimate_probabilities(L, estimator, n_additional_empty_bins)
    H_0 = n_additional_empty_bins * P_0 * -np.log2(P_0 + np.spacing(0)) / np.log2(base)
    H = entropy_pmf(P, base, require_valid_pmf=False) + H_0

    if keep_dims:
        H = H[..., np.newaxis]

    return H


def entropy_conditional(
    X,
    Y=None,
    cartesian_product=False,
    base=2,
    fill_value=-1,
    estimator="ML",
    Alphabet_X=None,
    Alphabet_Y=None,
    keep_dims=False,
):
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(
            Alphabet_X, fill_value
        )
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X, fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(
            Alphabet_Y, fill_value
        )
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y, fill_value_Y)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not " "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not " "match")
    if not cartesian_product and X.shape != Y.shape:
        raise ValueError("dimensions of args X and Y do not match")
    if cartesian_product and X.shape[-1] != Y.shape[-1]:
        raise ValueError("trailing dimensions of args X and Y do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers(
        (X, Alphabet_X, Y, Alphabet_Y),
        (fill_value_X, fill_value_Alphabet_X, fill_value_Y, fill_value_Alphabet_Y),
    )
    X, Alphabet_X, Y, Alphabet_Y = S

    if not cartesian_product:
        H = np.empty(X.shape[:-1])
        if H.ndim > 0:
            H[:] = np.NaN
        else:
            H = np.float64(np.NaN)
    else:

        def f(X, Y, Alphabet_X, Alphabet_Y):
            return entropy_conditional(
                X, Y, False, base, fill_value, estimator, Alphabet_X, Alphabet_Y
            )

        return _cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y)

    # Re-shape H, X and Y, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    for i in range(X.shape[0]):
        H[i] = entropy_joint(
            np.vstack((X[i], Y[i])),
            base,
            fill_value,
            estimator,
            _vstack_pad((Alphabet_X[i], Alphabet_Y[i]), fill_value),
        ) - entropy(Y[i], base, fill_value, estimator, Alphabet_Y[i])

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def entropy(X, base=2, fill_value=-1, estimator="ML", Alphabet_X=None, keep_dims=False):
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(
            Alphabet_X, fill_value
        )
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X, fill_value_X)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not " "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers(
        (X, Alphabet_X), (fill_value_X, fill_value_Alphabet_X)
    )
    X, Alphabet_X = S

    H = np.empty(X.shape[:-1])
    if H.ndim > 0:
        H[:] = np.NaN
    else:
        H = np.float64(np.NaN)

    # Re-shape H and X, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)

    # NB: This is not joint entropy. Elements in each row are sorted
    # independently
    X = np.sort(X, axis=1)

    # Compute symbol run-lengths
    # Compute symbol change indicators
    B = X[:, 1:] != X[:, :-1]
    for i in range(X.shape[0]):
        # Obtain symbol change positions
        I = np.append(np.where(B[i]), X.shape[1] - 1)
        # Compute run lengths
        L = np.diff(np.append(-1, I))

        alphabet_X = X[i, I]
        if estimator != "ML":
            n_additional_empty_bins = _determine_number_additional_empty_bins(
                L, alphabet_X, Alphabet_X[i], fill_value
            )
        else:
            n_additional_empty_bins = 0
        L, _ = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
        if not np.any(L):
            continue

        # P_0 is the probability mass assigned to each additional empty bin
        P, P_0 = _estimate_probabilities(L, estimator, n_additional_empty_bins)
        H_0 = (
            n_additional_empty_bins
            * P_0
            * -np.log2(P_0 + np.spacing(0))
            / np.log2(base)
        )
        H[i] = entropy_pmf(P, base, require_valid_pmf=False) + H_0

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims:
        H = H[..., np.newaxis]

    return H


def _autocreate_alphabet(X, fill_value):
    Lengths = np.apply_along_axis(lambda x: np.unique(x).size, axis=-1, arr=X)
    max_length = np.max(Lengths)

    def pad_with_fillvalue(x):
        return np.append(x, np.tile(fill_value, int(max_length - x.size)))

    Alphabet = np.apply_along_axis(
        lambda x: pad_with_fillvalue(np.unique(x)), axis=-1, arr=X
    )
    return (Alphabet, fill_value)


def _sanitise_array_input(x, fill_value=-1):
    # Avoid Python 3 issues with numpy arrays containing None elements
    if np.any(np.equal(x, None)) or fill_value is None:
        x = np.array(x, copy=False)
        assert np.all(x != NONE_REPLACEMENT)
        m = np.equal(x, None)
        x = np.where(m, NONE_REPLACEMENT, x)
    if fill_value is None:
        x = np.array(x, copy=False)
        fill_value = NONE_REPLACEMENT

    if isinstance(x, (pd.core.frame.DataFrame, pd.core.series.Series)):
        # Create masked array, honouring Dataframe/Series missing entries
        # NB: We transpose for convenience, so that quantities are computed for
        # each column
        x = np.ma.MaskedArray(x, x.isnull())

    if isinstance(x, np.ma.MaskedArray):
        fill_value = x.fill_value

        if np.any(x == fill_value):
            warnings.warn("Masked array contains data equal to fill value")

        if x.dtype.kind in ("S", "U"):
            kind = x.dtype.kind
            current_dtype_len = int(x.dtype.str.split(kind)[1])
            if current_dtype_len < len(fill_value):
                # Fix numpy's broken array string type behaviour which causes
                # X.filled() placeholder entries to be no longer than
                # non-placeholder entries
                warnings.warn(
                    "Changing numpy array dtype internally to "
                    "accommodate fill_value string length"
                )
                m = x.mask
                x = np.array(x.filled(), dtype=kind + str(len(fill_value)))
                x[m] = fill_value
            else:
                x = x.filled()
        else:
            x = x.filled()
    else:
        x = np.array(x, copy=False)

    if x.dtype.kind not in "biufcmMOSUV":
        raise TypeError("Unsupported array dtype")

    if x.size == 1 and x.ndim == 0:
        x = np.array((x,))

    return x, np.array(fill_value)


def _map_observations_to_integers(Symbol_matrices, Fill_values):
    assert len(Symbol_matrices) == len(Fill_values)
    FILL_VALUE = -1
    if np.any([A.dtype != "int" for A in Symbol_matrices]) or np.any(
        np.array(Fill_values) != FILL_VALUE
    ):
        L = sklearn.preprocessing.LabelEncoder()
        F = [np.atleast_1d(v) for v in Fill_values]
        L.fit(np.concatenate([A.ravel() for A in Symbol_matrices] + F))
        Symbol_matrices = [
            L.transform(A.ravel()).reshape(A.shape) for A in Symbol_matrices
        ]
        Fill_values = [L.transform(np.atleast_1d(f)) for f in Fill_values]

        for A, f in zip(Symbol_matrices, Fill_values):
            assert not np.any(A == FILL_VALUE)
            A[A == f] = FILL_VALUE

    assert np.all([A.dtype == "int" for A in Symbol_matrices])
    return Symbol_matrices, FILL_VALUE


def _isnan(X):
    X = np.array(X, copy=False)
    if X.dtype in ("int", "float"):
        return np.isnan(X)
    else:
        f = np.vectorize(_isnan_element)
        return f(X)


def _isnan_element(x):
    if isinstance(x, type(np.nan)):
        return np.isnan(x)
    else:
        return False


def _determine_number_additional_empty_bins(
    Counts, Alphabet, Full_Alphabet, fill_value
):
    alphabet_sizes = np.sum(np.atleast_2d(Full_Alphabet) != fill_value, axis=-1)
    if np.any(alphabet_sizes != fill_value):
        joint_alphabet_size = np.prod(alphabet_sizes[alphabet_sizes > 0])
        if joint_alphabet_size <= 0:
            raise ValueError(
                "Numerical overflow detected. Joint alphabet " "size too large."
            )
    else:
        joint_alphabet_size = 0
    return joint_alphabet_size - np.sum(
        np.all(np.atleast_2d(Alphabet) != fill_value, axis=0)
    )


def _estimate_probabilities(Counts, estimator, n_additional_empty_bins=0):
    # TODO Documentation should present the following guidelines:
    # 1) Good-Turing may be used if slope requirement satisfied and if
    # unobserved symbols have been defined (TODO Clarify what the requirement
    # is)
    # 2) James-Stein approach may be used as an alternative
    # 3) Dirichlet prior may be used in all other cases

    assert np.sum(Counts) > 0
    assert np.all(Counts.astype("int") == Counts)
    assert n_additional_empty_bins >= 0
    Counts = Counts.astype("int")

    if isinstance(estimator, str):
        estimator = estimator.upper().replace(" ", "")

    if np.isreal(estimator) or estimator in ("ML", "PERKS", "MINIMAX"):
        if np.isreal(estimator):
            alpha = estimator
        elif estimator == "PERKS":
            alpha = 1.0 / (Counts.size + n_additional_empty_bins)
        elif estimator == "MINIMAX":
            alpha = np.sqrt(np.sum(Counts)) / (Counts.size + n_additional_empty_bins)
        else:
            alpha = 0
        Theta = (Counts + alpha) / (
            1.0 * np.sum(Counts) + alpha * (Counts.size + n_additional_empty_bins)
        )
        # Theta_0 is the probability mass assigned to each additional empty bin
        if n_additional_empty_bins > 0:
            Theta_0 = alpha / (
                1.0 * np.sum(Counts) + alpha * (Counts.size + n_additional_empty_bins)
            )
        else:
            Theta_0 = 0

    elif estimator == "GOOD-TURING":
        # TODO We could also add a Chen-Chao vocabulary size estimator (See
        # Bhat Suma's thesis)

        # The following notation is based on Gale and Sampson (1995)
        # Determine histogram of counts N_r (index r denotes count)
        X = np.sort(Counts)
        B = X[1:] != X[:-1]  # Compute symbol change indicators
        I = np.append(np.where(B), X.size - 1)  # Obtain symbol change positions
        N_r = np.zeros(X[I[-1]] + 1)
        N_r[X[I]] = np.diff(np.append(-1, I))  # Compute run lengths
        N_r[0] = 0  # Ensures that unobserved symbols do not interfere

        # Compute Z_r, a locally averaged version of N_r
        R = np.where(N_r)[0]
        Q = np.append(0, R[:-1])
        T = np.append(R[1:], 2 * R[-1] - Q[-1])
        Z_r = np.zeros_like(N_r)
        Z_r[R] = N_r[R] / (0.5 * (T - Q))

        # Fit least squares regression line to plot of log(Z_r) versus log(r)
        x = np.log10(np.arange(1, Z_r.size))
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log10(Z_r[1:])
        x = x[np.isfinite(y)]
        y = y[np.isfinite(y)]
        m, c = np.linalg.lstsq(np.vstack([x, np.ones(x.size)]).T, y, rcond=None)[0]
        if m >= -1:
            warnings.warn(
                "Regression slope < -1 requirement in linear "
                "Good-Turing estimate not satisfied"
            )
        # Compute smoothed value of N_r based on interpolation
        # We need to refer to SmoothedN_{r+1} for all observed values of r
        SmoothedN_r = np.zeros(N_r.size + 1)
        SmoothedN_r[1:] = 10 ** (np.log10(np.arange(1, SmoothedN_r.size)) * m + c)

        # Determine threshold value of r at which to use smoothed values of N_r
        # (SmoothedN_r), as apposed to straightforward N_r.
        # Variance of Turing estimate
        with np.errstate(invalid="ignore", divide="ignore"):
            VARr_T = (
                (np.arange(N_r.size) + 1) ** 2
                * (1.0 * np.append(N_r[1:], 0) / (N_r**2))
                * (1 + np.append(N_r[1:], 0) / N_r)
            )
            x = (np.arange(N_r.size) + 1) * 1.0 * np.append(N_r[1:], 0) / N_r
            y = (np.arange(N_r.size) + 1) * 1.0 * SmoothedN_r[1:] / (SmoothedN_r[:-1])
            assert np.isinf(VARr_T[0]) or np.isnan(VARr_T[0])
            turing_is_sig_diff = np.abs(x - y) > 1.96 * np.sqrt(VARr_T)
        assert turing_is_sig_diff[0] == np.array(False)
        # NB: 0th element can be safely ignored, since always 0
        T = np.where(turing_is_sig_diff == np.array(False))[0]
        if T.size > 1:
            thresh_r = T[1]
            # Use smoothed estimates from the first non-significant
            # np.abs(SmoothedN_r-N_r) position onwards
            SmoothedN_r[:thresh_r] = N_r[:thresh_r]
        else:
            # Use only non-smoothed estimates (except for SmoothedN_r[-1])
            SmoothedN_r[:-1] = N_r

        # Estimate probability of encountering one particular symbol among the
        # objects observed r times, r>0
        p_r = np.zeros(N_r.size)
        N = np.sum(Counts)
        p_r[1:] = (
            (np.arange(1, N_r.size) + 1)
            * 1.0
            * SmoothedN_r[2:]
            / (SmoothedN_r[1:-1] * N)
        )
        # Estimate probability of observing any unseen symbol
        p_r[0] = 1.0 * N_r[1] / N

        # Assign probabilities to observed symbols
        Theta = np.array([p_r[r] for r in Counts])
        Theta[Counts == 0] = 0

        # Normalise probabilities for observed symbols, so that they sum to one
        if np.any(Counts == 0) or n_additional_empty_bins > 0:
            Theta = (1 - p_r[0]) * Theta / np.sum(Theta)
        else:
            warnings.warn(
                "No unobserved outcomes specified. Disregarding the "
                "probability mass allocated to any unobserved "
                "outcomes."
            )
            Theta = Theta / np.sum(Theta)

        # Divide p_0 among unobserved symbols
        with np.errstate(invalid="ignore", divide="ignore"):
            p_emptybin = p_r[0] / (np.sum(Counts == 0) + n_additional_empty_bins)
        Theta[Counts == 0] = p_emptybin
        # Theta_0 is the probability mass assigned to each additional empty bin
        if n_additional_empty_bins > 0:
            Theta_0 = p_emptybin
        else:
            Theta_0 = 0

    elif estimator == "JAMES-STEIN":
        Theta, _ = _estimate_probabilities(Counts, "ML")
        p_uniform = 1.0 / (Counts.size + n_additional_empty_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            Lambda = (1 - np.sum(Theta**2)) / (
                (np.sum(Counts) - 1)
                * (
                    np.sum((p_uniform - Theta) ** 2)
                    + n_additional_empty_bins * p_uniform**2
                )
            )

        if Lambda > 1:
            Lambda = 1
        elif Lambda < 0:
            Lambda = 0
        elif np.isnan(Lambda):
            Lambda = 1

        Theta = Lambda * p_uniform + (1 - Lambda) * Theta
        # Theta_0 is the probability mass assigned to each additional empty bin
        if n_additional_empty_bins > 0:
            Theta_0 = Lambda * p_uniform
        else:
            Theta_0 = 0
    else:
        raise ValueError("invalid value specified for estimator")

    return Theta, Theta_0


def _remove_counts_at_fill_value(Counts, Alphabet, fill_value):
    I = np.any(np.atleast_2d(Alphabet) == fill_value, axis=0)
    if np.any(I):
        Counts = Counts[~I]
        Alphabet = Alphabet.T[~I].T
    return (Counts, Alphabet)


def _cartesian_product_apply(X, Y, function, Alphabet_X=None, Alphabet_Y=None):
    assert X.ndim > 0 and Y.ndim > 0
    assert X.size > 0 and Y.size > 0
    if Alphabet_X is not None or Alphabet_Y is not None:
        assert Alphabet_X.ndim > 0 and Alphabet_Y.ndim > 0
        assert Alphabet_X.size > 0 and Alphabet_Y.size > 0

    H = np.empty(np.append(X.shape[:-1], Y.shape[:-1]).astype("int"))
    if H.ndim > 0:
        H[:] = np.NaN
    else:
        H = np.float64(np.NaN)

    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    if Alphabet_X is not None or Alphabet_Y is not None:
        Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
        Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    n = 0
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            if Alphabet_X is not None or Alphabet_Y is not None:
                H[n] = function(X[i], Y[j], Alphabet_X[i], Alphabet_Y[j])
            else:
                H[n] = function(X[i], Y[j])
            n = n + 1

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    return H


def _verify_alphabet_sufficiently_large(X, Alphabet, fill_value):
    assert not np.any(X == np.array(None))
    assert not np.any(Alphabet == np.array(None))
    for i in range(X.shape[0]):
        I = X[i] != fill_value
        J = Alphabet[i] != fill_value
        # NB: This causes issues when both arguments contain None. But it is
        # always called after observations have all been mapped to integers.
        if np.setdiff1d(X[i, I], Alphabet[i, J]).size > 0:
            raise ValueError(
                "provided alphabet does not contain all observed " "outcomes"
            )


def _vstack_pad(arrays, fill_value):
    max_length = max([A.shape[-1] for A in arrays])
    arrays = [
        np.append(
            A,
            np.tile(
                fill_value,
                np.append(A.shape[:-1], max_length - A.shape[-1]).astype(int),
            ),
        )
        for A in arrays
    ]
    return np.vstack(arrays)
