import numpy as np

def generate_partitions(X, T, n_folds, validation=True,
                        shuffle=True, classification=True):
    '''Returns Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
      or
       Xtrain,Ttrain,Xtest,Ttest if validation is False
    Build dictionary keyed by class label. Each entry contains rowIndices and start and stop
    indices into rowIndices for each of n_folds folds'''
    global folds
    if not classification:
        print('Not implemented yet.')
        return
    row_indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(row_indices)
    folds = {}
    classes = np.unique(T)
    for c in classes:
        class_indices = row_indices[np.where(T[row_indices, :] == c)[0]]
        n_in_class = len(class_indices)
        n_each = int(n_in_class / n_folds)
        starts = np.arange(0, n_each * n_folds, n_each)
        stops = starts + n_each
        stops[-1] = n_in_class
        # startsStops = np.vstack((row_indices[starts],row_indices[stops])).T
        folds[c] = [class_indices, starts, stops]

    for test_fold in range(n_folds):
        if validation:
            for validate_fold in range(n_folds):
                if test_fold == validate_fold:
                    continue
                train_folds = np.setdiff1d(range(n_folds), [test_fold,validate_fold])
                rows = rows_in_fold(folds,test_fold)
                Xtest = X[rows, :]
                Ttest = T[rows, :]
                rows = rows_in_fold(folds,validate_fold)
                Xvalidate = X[rows, :]
                Tvalidate = T[rows, :]
                rows = rows_in_folds(folds,train_folds)
                Xtrain = X[rows, :]
                Ttrain = T[rows, :]
                yield Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest
        else:
            # No validation set
            train_folds = np.setdiff1d(range(n_folds), [test_fold])
            rows = rows_in_fold(folds,test_fold)
            Xtest = X[rows, :]
            Ttest = T[rows, :]
            rows = rows_in_folds(folds,train_folds)
            Xtrain = X[rows, :]
            Ttrain = T[rows, :]
            yield Xtrain, Ttrain, Xtest, Ttest
            

def rows_in_fold(folds, k):
    allRows = []
    for c,rows in folds.items():
        classRows, starts, stops = rows
        allRows += classRows[starts[k]:stops[k]].tolist()
    return allRows

def rows_in_folds(folds, ks):
    allRows = []
    for k in ks:
        allRows += rows_in_fold(folds, k)
    return allRows



def list_to_tuple(lst):
    return tuple(list_to_tuple(x) for x in lst) if isinstance(lst, list) else lst


def zip_longest(lists):
    def g(l):
        for item in l:
            yield item
        while True:
            yield None
    gens = [g(l) for l in lists]    
    order = []
    for _ in range(max(map(len, lists))):
        for g in gens:
            nextg = next(g)
            if nextg is not None:
                order.append(nextg)
    return order
        # yield tuple(next(g) for g in gens)

def order_for_stratified_sampling(T):
    uniques = np.unique(T)
    order = np.arange(T.shape[0])
    orders = []
    for unique in uniques:
        orders.append(np.where(T == unique)[0].tolist())
    # print(orders)
    new_order = zip_longest(orders)
    # print(new_order)
    return new_order
    # return np.array(orders).T.ravel()

def make_batches(X, T, batch_size):
    
    if batch_size == -1:
        yield X, T
    else:
        n_samples = X.shape[0]
        new_order = np.arange(n_samples)
        np.random.shuffle(new_order)
        X = X[new_order, ...]
        T = T[new_order, ...]
        new_order = order_for_stratified_sampling(T)
        for batch_start in range(0, n_samples, batch_size):
            batch_end = batch_start + batch_size
            # print(f'    {batch_start=} {batch_end=}')
            batch_rows = new_order[batch_start:batch_end]
            # print(np.sum(T[batch_rows, ...], axis=0))
            yield X[batch_rows, ...], T[batch_rows, ...]


def percent_equal(Y, T):
    return 100 * np.mean(Y == T)
    # return 100 * np.mean(np.equal(Y, T))

        
