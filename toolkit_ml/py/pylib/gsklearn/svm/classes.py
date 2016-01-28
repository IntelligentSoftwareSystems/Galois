# Donald Nguyen <ddn@cs.utexas.edu>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator
from ..preprocessing import LabelEncoder
from ..linear_model.base import LinearClassifierMixin, SparseCoefMixin
from ..feature_selection.from_model import _LearntSelectorMixin 
from ..utils import check_X_y, compute_class_weight, check_random_state, ConvergenceWarning

from toolkit_ml import train_linear


class LinearSVC(BaseEstimator, LinearClassifierMixin,
                _LearntSelectorMixin, SparseCoefMixin):
    """Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better (to large numbers of
    samples).

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    loss : string, 'l1' or 'l2' (default='l2')
        Specifies the loss function. 'l1' is the hinge loss (standard SVM)
        while 'l2' is the squared hinge loss.

    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to `coef_`
        vectors that are sparse.

    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria

    multi_class: string, 'ovr' or 'crammer_singer' (default='ovr')
        Determines the multi-class strategy if `y` contains more than
        two classes.
        `ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`
        optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from an theoretical perspective
        as it is consistent it is seldom used in practice and rarely leads to
        better accuracy and is more expensive to compute.
        If `crammer_singer` is chosen, the options loss, penalty and dual will
        be ignored.

    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    intercept_scaling : float, optional (default=1)
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.

    verbose : int, default: 0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    max_iter : int, default 1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : array, shape = [n_features] if n_classes == 2 \
            else [n_classes, n_features]
        Weights asigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    The underlying implementation (liblinear) uses a sparse internal
    representation for the data that will incur a memory copy.

    **References:**
    `LIBLINEAR: A Library for Large Linear Classification
    <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__

    See also
    --------
    SVC
        Implementation of Support Vector Machine classifier using libsvm:
        the kernel can be non-linear but its SMO algorithm does not
        scale to large number of samples as LinearSVC does.

        Furthermore SVC multi-class mode is implemented using one
        vs one scheme while LinearSVC uses one vs the rest. It is
        possible to implement one vs the rest with SVC by using the
        :class:`sklearn.multiclass.OneVsRestClassifier` wrapper.

        Finally SVC can fit dense data without memory copy if the input
        is C-contiguous. Sparse data will still incur memory copy though.

    sklearn.linear_model.SGDClassifier
        SGDClassifier can optimize the same cost function as LinearSVC
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.

    """

    def __init__(self, penalty='l2', loss='l2', dual=True, tol=1e-4, C=1.0,
                multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                class_weight=None, verbose=0, random_state=None, max_iter=1000):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = 1000 if max_iter < 0 else max_iter

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        Returns
        -------
        self : object
            Returns self.
        """
        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64, order="C")
        self.classes_ = np.unique(y)
        
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        classes_ = enc.classes_
        if len(classes_) < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])
        class_weight_ = compute_class_weight(self.class_weight, classes_, y)
        rnd = check_random_state(self.random_state)
        if self.verbose:
            print('[LibLinear]', end='')
        
        bias = -1.0
        if self.fit_intercept:
            bias = self.fit_intercept

        # LibLinear wants targets as doubles even for classification
        y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
        raw_coef_, n_iter_ = train_linear(X, y_ind,
                self.loss, self.penalty, self.dual, self.tol, bias, self.C, 
                class_weight_, self.max_iter, rnd.randint(np.iinfo('i').max))

        self.n_iter_ = max(n_iter_)
        if self.n_iter_ >= self.max_iter and self.verbose > 0:
            warnings.warn("Liblinear failed to converge, increase "
                          "the number of iterations.", ConvergenceWarning)
        
        if self.fit_intercept:
            self.coef_ = raw_coef_[:, :-1]
            self.intercept_ = self.intercept_scaling * raw_coef_[:, -1]
        else:
            self.coef = raw_coef_
            self.intercept_ = 0

        # TODO
        #self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
        #    X, y, self.C, self.fit_intercept, self.intercept_scaling,
        #    self.class_weight, self.penalty, self.dual, self.verbose,
        #    self.max_iter, self.tol, self.random_state, self.multi_class,
        #    self.loss
        #    )

        if self.multi_class == "crammer_singer" and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0]).reshape(1, -1)
            if self.fit_intercept:
                intercept = self.intercept_[1] - self.intercept_[0]
                self.intercept_ = np.array([intercept])

        return self


class SVC(LinearSVC):
    """C-Support Vector Classification.

    The implementations is a based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to dataset with more than a couple of 10000 samples.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each,
    see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.

    .. The narrative documentation is available at http://scikit-learn.org/

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 0.0 then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    probability: boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.

    shrinking: boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB)

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data for probability estimation.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Index of support vectors.

    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.

    n_support_ : array-like, dtype=int32, shape = [n_class]
        number of support vector for each class.

    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function. \
        For multiclass, coefficient for all 1-vs-1 classifiers. \
        The layout of the coefficients in the multiclass case is somewhat \
        non-trivial. See the section about multi-class classification in the \
        SVM section of the User Guide for details.

    coef_ : array, shape = [n_class-1, n_features]
        Weights asigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`

    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = SVC()
    >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    SVR
        Support Vector Machine for Regression implemented using libsvm.

    LinearSVC
        Scalable Linear Support Vector Machine for classification
        implemented using liblinear. Check the See also section of
        LinearSVC for more comparison element.

    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, random_state=None):
        if kernel != 'linear':
            raise ValueError("Only linear kernel is supported")
        if probability:
            raise ValueError("probability not yet supported")
        super(SVC, self).__init__(C=C, tol=tol, class_weight=class_weight,
                verbose=1 if verbose else 0, max_iter=max_iter, random_state=None)

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        # TODO
        pass

    @property
    def predict_log_proba(self):
        """Compute log probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probabilities of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        # TODO
        pass

# vim: set ts=4 sw=4:
