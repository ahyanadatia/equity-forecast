"""A recurrent network over the lagged feature frame - the one non-linear model here.

Ridge reads a single feature row and maps it linearly to a forecast. The LSTM is
given the last ``sequence_length`` feature *rows* of the same ticker and is free
to learn an interaction across time that a flat regression cannot express (a
short burst of volatility followed by a quiet day is a different state from the
reverse ordering, even though both average to the same thing).

Everything about the design is chosen to keep it honest rather than to make the
in-sample loss small:

*The sequence never reaches forward.* The feature frame is already constructed so
that row ``t`` contains only information observable at the close of ``t-1``
(see :mod:`equity_forecast.features`). A sequence ending at row ``t`` is built
from rows ``t - sequence_length + 1 .. t`` **of that ticker alone**, so the
forecast for ``t`` is a function of rows ``<= t`` and nothing else. Rows near the
start of the prediction window have no full history behind them; they are given
the training-set mean rather than a window silently zero-padded (which would
invent data) or one borrowed from a neighbouring ticker (which would leak the
cross-section in through the back door).

*Early stopping validates on the future, not on a shuffle.* The validation slice
is the chronologically **last** ``val_frac`` of the training fold's dates. A
random split would put days from the middle of the training period into the
validation set while their neighbours - which share overlapping rolling windows
and sit inside the same volatility regime - stay in the training set. The
validation loss would then measure interpolation between known days rather than
extrapolation to unknown ones, early stopping would fire far too late, and the
selected epoch would be optimistic in exactly the direction that matters. Slicing
by time reproduces the deployment problem: fit on the past, stop on the most
recent stretch, and never look at anything from the embargo gap or the test block.

*Pooled by default.* One network is fitted across the whole cross-section with a
learned ticker embedding concatenated to the recurrent output, so shared dynamics
are estimated on ``n_tickers`` times as much data while the embedding still gives
each name its own offset. Index ``0`` of the embedding table is reserved for a
ticker that was never seen during training, which is what makes the model safe to
call on a window containing a new name. Setting ``pooled=False`` fits one
independent network per ticker instead.

*Determinism.* :func:`equity_forecast.seeding.set_global_seeds` is called at the
top of :meth:`LstmForecaster.fit`, every initializer and dropout layer carries an
explicit seed, and ``model.fit`` runs with ``shuffle=False``. Two fits on the same
data with the same seed produce bitwise-identical forecasts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from equity_forecast.config import LstmConfig
from equity_forecast.models.base import BaseForecaster, register_model
from equity_forecast.models.naive import _require_fitted
from equity_forecast.seeding import set_global_seeds

__all__ = ["LstmForecaster", "chronological_validation_split"]

LOGGER = logging.getLogger(__name__)

_UNKNOWN_TICKER = 0
"""Embedding row reserved for tickers absent from the training fold."""

_KERAS: Any = None


def _keras() -> Any:  # noqa: ANN401 - the keras module has no useful static type here
    """Import Keras lazily, keeping TensorFlow's start-up chatter off stderr.

    TensorFlow costs several seconds to import and is an optional extra, so it is
    loaded on first use rather than when this module is imported. The result is
    memoised because repeated imports are cheap but not free.

    Returns:
        The ``keras`` module bundled with TensorFlow.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    global _KERAS  # noqa: PLW0603 - a module-level import cache is exactly what this is
    if _KERAS is None:
        import os  # noqa: PLC0415 - must be set before the TensorFlow C++ runtime starts

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf  # noqa: PLC0415 - optional dependency, imported lazily

        tf.get_logger().setLevel(logging.ERROR)
        _KERAS = tf.keras
    return _KERAS


def chronological_validation_split(
    dates: pd.Index, val_frac: float
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a set of training dates into a past block and a future validation block.

    The validation block is the **last** ``val_frac`` of the distinct dates in
    time order - never a random sample. See the module docstring for why that
    distinction decides whether early stopping means anything.

    The count is rounded down, so ``val_frac=0.15`` over 100 dates validates on
    15 of them. At least one date is always left for training, so a degenerate
    ``val_frac`` cannot empty the training set.

    Args:
        dates: Dates of the training fold. Duplicates (one row per ticker per
            date) and unsorted input are both fine; they are reduced to the
            sorted unique dates first.
        val_frac: Fraction of the distinct dates to hold out, in ``[0, 1]``.
            Values ``<= 0`` produce an empty validation block.

    Returns:
        ``(train_dates, val_dates)``: two disjoint, sorted ``DatetimeIndex``
        objects whose union is the distinct dates of ``dates``, with every entry
        of ``train_dates`` strictly earlier than every entry of ``val_dates``.

    Raises:
        ValueError: If ``val_frac`` is not in ``[0, 1]``.
    """
    if not 0.0 <= val_frac <= 1.0:
        msg = f"val_frac must lie in [0, 1], got {val_frac}"
        raise ValueError(msg)
    unique = pd.DatetimeIndex(pd.Index(dates).unique()).sort_values()
    n = len(unique)
    if n < 2 or val_frac <= 0.0:
        return unique, unique[:0]
    # Floor with a tolerance, so 0.15 * 20 landing on 2.9999999999999996 still gives 3.
    n_val = int(n * val_frac + 1e-9)
    n_val = min(max(n_val, 0), n - 1)
    return unique[: n - n_val], unique[n - n_val :]


def _ticker_level(index: pd.Index) -> npt.NDArray[np.object_]:
    """Return the ``ticker`` level of a dataset index as an object array.

    Args:
        index: A ``(date, ticker)`` MultiIndex.

    Returns:
        One ticker label per row.

    Raises:
        KeyError: If ``index`` has no ``ticker`` level.
    """
    if not isinstance(index, pd.MultiIndex) or "ticker" not in (index.names or []):
        msg = "lstm requires a (date, ticker) MultiIndex; no 'ticker' level found"
        raise KeyError(msg)
    return np.asarray(index.get_level_values("ticker"), dtype=object)


def _date_level(index: pd.Index) -> npt.NDArray[np.datetime64]:
    """Return the ``date`` level of a dataset index as a datetime64 array.

    Args:
        index: A ``(date, ticker)`` MultiIndex.

    Returns:
        One date per row.

    Raises:
        KeyError: If ``index`` has no ``date`` level.
    """
    if not isinstance(index, pd.MultiIndex) or "date" not in (index.names or []):
        msg = "lstm requires a (date, ticker) MultiIndex; no 'date' level found"
        raise KeyError(msg)
    return np.asarray(index.get_level_values("date"), dtype="datetime64[ns]")


def _imputed_matrix(
    x: pd.DataFrame, columns: list[str], impute: npt.NDArray[np.float64]
) -> npt.NDArray[np.float32]:
    """Select the fitted feature columns and replace non-finite cells.

    Args:
        x: Feature frame.
        columns: Columns to select, in the order the model was fitted on.
        impute: Replacement value per column, computed on the training fold.

    Returns:
        A ``float32`` matrix with no non-finite entries.
    """
    matrix = np.asarray(x[columns].to_numpy(dtype="float64"), dtype="float64").copy()
    bad = ~np.isfinite(matrix)
    if bad.any():
        LOGGER.debug("lstm: imputing %d non-finite feature cell(s)", int(bad.sum()))
        matrix[bad] = np.take(impute, np.nonzero(bad)[1])
    return matrix.astype("float32")


def _window_sequences(
    matrix: npt.NDArray[np.float32],
    tickers: npt.NDArray[np.object_],
    dates: npt.NDArray[np.datetime64],
    sequence_length: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
    """Build one backward-looking input sequence per row that has a full history.

    For every ticker independently, the rows are put in date order and a sliding
    window of length ``sequence_length`` is taken over them. The window at output
    position ``i`` covers rows ``i .. i + sequence_length - 1`` of that ticker and
    is attached to the **last** of those rows, so a sequence never contains a row
    dated after the row it predicts. Rows in the first ``sequence_length - 1``
    positions of a ticker get no sequence at all and are reported as missing.

    Args:
        matrix: ``(n_rows, n_features)`` feature matrix, free of non-finite cells.
        tickers: Ticker label of each row of ``matrix``.
        dates: Date of each row of ``matrix``.
        sequence_length: Number of rows in each window.

    Returns:
        ``(sequences, positions)`` where ``sequences`` has shape
        ``(m, sequence_length, n_features)`` and ``positions[i]`` is the row of
        ``matrix`` that ``sequences[i]`` ends on and therefore predicts. Both are
        empty when no ticker has enough rows.
    """
    n_features = matrix.shape[1]
    seq_blocks: list[npt.NDArray[np.float32]] = []
    pos_blocks: list[npt.NDArray[np.intp]] = []

    for ticker in np.unique(tickers):
        rows = np.flatnonzero(tickers == ticker)
        rows = rows[np.argsort(dates[rows], kind="stable")]
        if rows.size < sequence_length:
            continue
        block = matrix[rows]
        # (n_t - L + 1, n_features, L) -> (n_t - L + 1, L, n_features)
        windows = np.lib.stride_tricks.sliding_window_view(block, sequence_length, axis=0)
        seq_blocks.append(np.ascontiguousarray(windows.transpose(0, 2, 1), dtype="float32"))
        pos_blocks.append(np.asarray(rows[sequence_length - 1 :], dtype=np.intp))

    if not seq_blocks:
        empty_seq = np.zeros((0, sequence_length, n_features), dtype="float32")
        return empty_seq, np.zeros(0, dtype=np.intp)
    return np.concatenate(seq_blocks, axis=0), np.concatenate(pos_blocks, axis=0)


@register_model("lstm")
class LstmForecaster(BaseForecaster):
    """Keras LSTM over the last ``sequence_length`` feature rows of each ticker.

    Attributes:
        params: The :class:`~equity_forecast.config.LstmConfig` in force.
        seed: Seed applied to Python, NumPy and TensorFlow before training, and
            to every weight initializer and dropout layer.
    """

    def __init__(self, params: LstmConfig, seed: int) -> None:
        """Store configuration; no TensorFlow work happens here.

        Args:
            params: LSTM hyperparameters.
            seed: Random seed, used for global seeding and per-layer initializers.
        """
        self.params = params
        self.seed = seed
        self._columns: list[str] = []
        self._impute: npt.NDArray[np.float64] = np.zeros(0, dtype="float64")
        self._vocab: dict[str, int] = {}
        self._models: dict[str, Any] = {}
        self._pooled_model: Any = None
        self._global_mean: float = 0.0
        self._ticker_means: dict[str, float] = {}
        self._val_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._fitted = False

    # ----------------------------------------------------------------- fitting

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LstmForecaster:
        """Fit the network on the fold's training rows.

        Args:
            x: Scaled training feature frame indexed by ``(date, ticker)``.
            y: Training targets, aligned row-for-row with ``x``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``x`` is empty, has no columns, or ``y`` does not
                line up with ``x``.
            KeyError: If ``x`` is not indexed by ``(date, ticker)``.
        """
        set_global_seeds(self.seed)
        self._validate_training_frame(x, y)

        self._columns = [str(c) for c in x.columns]
        self._impute = _column_means(np.asarray(x.to_numpy(dtype="float64"), dtype="float64"))

        tickers = _ticker_level(x.index)
        dates = _date_level(x.index)
        target = np.asarray(y.to_numpy(), dtype="float64")
        self._record_training_means(target, tickers)
        self._vocab = {str(t): i + 1 for i, t in enumerate(sorted({str(t) for t in tickers}))}

        self._models = {}
        self._pooled_model = None
        self._fitted = True

        length = max(int(self.params.sequence_length), 1)
        matrix = _imputed_matrix(x, self._columns, self._impute)
        sequences, positions = _window_sequences(matrix, tickers, dates, length)
        if sequences.shape[0] == 0:
            LOGGER.warning(
                "lstm.fit(): no ticker has %d consecutive training rows (largest has %d); "
                "falling back to the training mean",
                length,
                int(pd.Series(tickers).value_counts().max()) if len(tickers) else 0,
            )
            return self

        usable = np.isfinite(target[positions])
        if not usable.all():
            LOGGER.debug(
                "lstm.fit(): dropping %d sequence(s) with a non-finite target", int((~usable).sum())
            )
            sequences, positions = sequences[usable], positions[usable]
        if sequences.shape[0] == 0:
            LOGGER.warning("lstm.fit(): every training sequence had a non-finite target")
            return self

        _, self._val_dates = chronological_validation_split(
            pd.DatetimeIndex(dates), float(self.params.val_frac)
        )
        is_val = np.isin(dates[positions], self._val_dates.to_numpy())

        if self.params.pooled:
            self._fit_pooled(sequences, positions, target, tickers, is_val)
        else:
            self._fit_per_ticker(sequences, positions, target, tickers, is_val)
        return self

    def _validate_training_frame(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Reject training input the model cannot use.

        Args:
            x: Training feature frame.
            y: Training targets.

        Raises:
            ValueError: If ``x`` is empty, has no columns, or ``y`` is misaligned.
        """
        if x.shape[1] == 0:
            msg = "lstm.fit() got a feature frame with no columns"
            raise ValueError(msg)
        if len(x) == 0:
            msg = "lstm.fit() got an empty training set"
            raise ValueError(msg)
        if len(y) != len(x):
            msg = f"lstm.fit() got {len(x)} feature rows but {len(y)} targets"
            raise ValueError(msg)

    def _record_training_means(
        self, target: npt.NDArray[np.float64], tickers: npt.NDArray[np.object_]
    ) -> None:
        """Remember the fallback forecasts: the training mean, overall and per ticker.

        Args:
            target: Training targets.
            tickers: Ticker label of each training row.
        """
        finite = np.isfinite(target)
        self._global_mean = float(target[finite].mean()) if finite.any() else 0.0
        self._ticker_means = {}
        for ticker in np.unique(tickers):
            mask = (tickers == ticker) & finite
            if mask.any():
                self._ticker_means[str(ticker)] = float(target[mask].mean())

    def _fit_pooled(
        self,
        sequences: npt.NDArray[np.float32],
        positions: npt.NDArray[np.intp],
        target: npt.NDArray[np.float64],
        tickers: npt.NDArray[np.object_],
        is_val: npt.NDArray[np.bool_],
    ) -> None:
        """Fit one network across every ticker, with a ticker embedding.

        Args:
            sequences: All training sequences.
            positions: Row of the training frame each sequence predicts.
            target: Training targets, indexed by training-frame row.
            tickers: Ticker label of each training-frame row.
            is_val: Whether each sequence falls in the chronological validation block.
        """
        codes = np.array(
            [self._vocab.get(str(t), _UNKNOWN_TICKER) for t in tickers[positions]],
            dtype="int32",
        ).reshape(-1, 1)
        model = self._build_network(sequences.shape[2], pooled=True)
        self._fit_network(
            model,
            {"sequence": sequences, "ticker": codes},
            target[positions].astype("float32"),
            is_val,
            label="pooled",
        )
        self._pooled_model = model

    def _fit_per_ticker(
        self,
        sequences: npt.NDArray[np.float32],
        positions: npt.NDArray[np.intp],
        target: npt.NDArray[np.float64],
        tickers: npt.NDArray[np.object_],
        is_val: npt.NDArray[np.bool_],
    ) -> None:
        """Fit one independent network per ticker.

        Args:
            sequences: All training sequences.
            positions: Row of the training frame each sequence predicts.
            target: Training targets, indexed by training-frame row.
            tickers: Ticker label of each training-frame row.
            is_val: Whether each sequence falls in the chronological validation block.
        """
        labels = tickers[positions]
        for offset, ticker in enumerate(sorted({str(t) for t in labels})):
            mask = labels == ticker
            if not mask.any():  # pragma: no cover - defensive
                continue
            set_global_seeds(self.seed + offset)
            model = self._build_network(sequences.shape[2], pooled=False)
            self._fit_network(
                model,
                sequences[mask],
                target[positions][mask].astype("float32"),
                is_val[mask],
                label=ticker,
            )
            self._models[ticker] = model
        missing = sorted(set(self._ticker_means) - set(self._models))
        if missing:
            LOGGER.warning(
                "lstm.fit(): tickers %s have fewer than %d training rows; "
                "they fall back to the training mean",
                missing,
                self.params.sequence_length,
            )

    def _build_network(self, n_features: int, *, pooled: bool) -> Any:  # noqa: ANN401
        """Assemble and compile the Keras model.

        A dropout layer follows every LSTM layer, including the last one, so the
        regulariser is still active in the default single-layer configuration.

        Args:
            n_features: Width of one feature row.
            pooled: Whether to add the ticker-embedding branch.

        Returns:
            A compiled ``keras.Model`` taking a ``sequence`` input (plus a
            ``ticker`` input when pooled) and emitting one linear output.
        """
        keras = _keras()
        length = max(int(self.params.sequence_length), 1)
        n_layers = max(int(self.params.n_layers), 1)
        units = max(int(self.params.units), 1)

        seq_input = keras.Input(shape=(length, n_features), dtype="float32", name="sequence")
        hidden = seq_input
        for i in range(n_layers):
            hidden = keras.layers.LSTM(
                units,
                return_sequences=i < n_layers - 1,
                kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed + 101 * i + 1),
                recurrent_initializer=keras.initializers.Orthogonal(seed=self.seed + 101 * i + 2),
                bias_initializer="zeros",
                name=f"lstm_{i + 1}",
            )(hidden)
            if self.params.dropout > 0.0:
                hidden = keras.layers.Dropout(
                    float(self.params.dropout),
                    seed=self.seed + 101 * i + 3,
                    name=f"dropout_{i + 1}",
                )(hidden)

        inputs = [seq_input]
        if pooled:
            tic_input = keras.Input(shape=(1,), dtype="int32", name="ticker")
            embedded = keras.layers.Embedding(
                input_dim=len(self._vocab) + 1,
                output_dim=max(int(self.params.embedding_dim), 1),
                embeddings_initializer=keras.initializers.RandomUniform(
                    -0.05, 0.05, seed=self.seed + 97
                ),
                name="ticker_embedding",
            )(tic_input)
            embedded = keras.layers.Flatten(name="ticker_embedding_flat")(embedded)
            hidden = keras.layers.Concatenate(name="state_and_ticker")([hidden, embedded])
            inputs.append(tic_input)

        output = keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed + 89),
            bias_initializer="zeros",
            name="head",
        )(hidden)
        model = keras.Model(inputs=inputs, outputs=output, name="lstm_forecaster")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(self.params.learning_rate)),
            loss="mse",
        )
        return model

    def _fit_network(
        self,
        model: Any,  # noqa: ANN401
        inputs: Any,  # noqa: ANN401
        targets: npt.NDArray[np.float32],
        is_val: npt.NDArray[np.bool_],
        *,
        label: str,
    ) -> None:
        """Train one compiled model with chronological early stopping.

        Args:
            model: The compiled Keras model.
            inputs: Either a sequence array or a ``{"sequence", "ticker"}`` dict.
            targets: Training targets, one per sequence.
            is_val: Whether each sequence belongs to the validation block.
            label: Name used in log messages ("pooled" or a ticker).
        """
        keras = _keras()
        train_mask = ~is_val
        if not train_mask.any():
            LOGGER.warning(
                "lstm.fit(): the chronological validation slice swallowed every %s sequence; "
                "training on all of them without early stopping",
                label,
            )
            train_mask = np.ones_like(is_val, dtype=bool)
            is_val = np.zeros_like(is_val, dtype=bool)

        callbacks: list[Any] = []
        validation_data: Any = None
        if is_val.any():
            validation_data = (_take(inputs, is_val), targets[is_val])
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=max(int(self.params.patience), 1),
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        else:
            LOGGER.debug("lstm.fit(): no validation sequences for %s; early stopping is off", label)

        model.fit(
            _take(inputs, train_mask),
            targets[train_mask],
            validation_data=validation_data,
            epochs=max(int(self.params.epochs), 1),
            batch_size=max(int(self.params.batch_size), 1),
            shuffle=False,  # NFR-1: a shuffled epoch order would break bitwise reproducibility.
            callbacks=callbacks,
            verbose=0,
        )

    # -------------------------------------------------------------- prediction

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict the target for every row of the fold's full window.

        Args:
            x: Scaled feature frame covering the fold's train rows, embargo gap
                and test rows, indexed by ``(date, ticker)``.

        Returns:
            A ``float64`` series indexed exactly like ``x``, named after the
            model's registry key. Rows without ``sequence_length`` prior rows for
            their own ticker carry the training mean.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If a feature column seen during fit is missing.
            KeyError: If ``x`` is not indexed by ``(date, ticker)``.
        """
        _require_fitted(self._fitted, self.name)
        missing = [c for c in self._columns if c not in x.columns]
        if missing:
            msg = f"lstm.predict() is missing feature column(s) seen during fit: {missing}"
            raise ValueError(msg)

        tickers = _ticker_level(x.index)
        dates = _date_level(x.index)
        values = np.array(
            [self._ticker_means.get(str(t), self._global_mean) for t in tickers], dtype="float64"
        )

        if self._pooled_model is not None or self._models:
            length = max(int(self.params.sequence_length), 1)
            matrix = _imputed_matrix(x, self._columns, self._impute)
            sequences, positions = _window_sequences(matrix, tickers, dates, length)
            if sequences.shape[0]:
                self._fill_predictions(values, sequences, positions, tickers)

        return pd.Series(values, index=x.index, name=self.name, dtype="float64")

    def _fill_predictions(
        self,
        values: npt.NDArray[np.float64],
        sequences: npt.NDArray[np.float32],
        positions: npt.NDArray[np.intp],
        tickers: npt.NDArray[np.object_],
    ) -> None:
        """Overwrite the mean fallback with network output wherever a sequence exists.

        Args:
            values: Output array, pre-filled with the training mean; modified in place.
            sequences: Sequences built from the prediction window.
            positions: Row of the window each sequence predicts.
            tickers: Ticker label of each window row.
        """
        batch = max(int(self.params.batch_size), 1)
        if self._pooled_model is not None:
            codes = np.array(
                [self._vocab.get(str(t), _UNKNOWN_TICKER) for t in tickers[positions]],
                dtype="int32",
            ).reshape(-1, 1)
            raw = self._pooled_model.predict(
                {"sequence": sequences, "ticker": codes}, batch_size=batch, verbose=0
            )
            values[positions] = np.asarray(raw, dtype="float64").ravel()
            return

        labels = tickers[positions]
        for ticker, model in self._models.items():
            mask = labels == ticker
            if not mask.any():
                continue
            raw = model.predict(sequences[mask], batch_size=batch, verbose=0)
            values[positions[mask]] = np.asarray(raw, dtype="float64").ravel()

    # ------------------------------------------------------------ introspection

    @property
    def validation_dates_(self) -> pd.DatetimeIndex:
        """The chronological validation slice used by early stopping.

        Returns:
            The training dates held out for early stopping, empty when the model
            fell back to the training mean or ``val_frac`` was zero.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return self._val_dates

    @property
    def ticker_vocabulary_(self) -> dict[str, int]:
        """The embedding index assigned to each training ticker.

        Returns:
            A mapping from ticker to embedding row, starting at ``1``; row ``0``
            is reserved for unseen tickers.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return dict(self._vocab)

    @property
    def uses_network_(self) -> bool:
        """Whether an actual network was trained.

        Returns:
            ``False`` when the training fold was too short to form a single
            sequence and the model degenerated to the training mean.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return self.n_networks_ > 0

    @property
    def n_networks_(self) -> int:
        """How many Keras models were actually trained.

        Returns:
            ``1`` in pooled mode, one per ticker with enough history when
            ``pooled=False``, and ``0`` when the training fold was too short to
            form a single sequence and the model degenerated to the training mean.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        if self._pooled_model is not None:
            return 1
        return len(self._models)


def _column_means(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Mean of the finite entries of each column, ``0.0`` for all-missing columns.

    Args:
        matrix: Training feature matrix.

    Returns:
        One imputation value per column.
    """
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0.0).sum(axis=0)
    return np.asarray(np.where(counts > 0, sums / np.maximum(counts, 1), 0.0), dtype="float64")


def _take(inputs: Any, mask: npt.NDArray[np.bool_]) -> Any:  # noqa: ANN401
    """Select rows of a model input, which may be an array or a dict of arrays.

    Args:
        inputs: A sequence array, or a mapping of input name to array.
        mask: Boolean row mask.

    Returns:
        The same structure, restricted to the masked rows.
    """
    if isinstance(inputs, dict):
        return {k: v[mask] for k, v in inputs.items()}
    return inputs[mask]
