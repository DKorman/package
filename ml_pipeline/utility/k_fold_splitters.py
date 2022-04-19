from sklearn.model_selection import KFold


def generate_non_chronological_year_folds(df):
    folds = []

    # get years for iteration purposes
    years = df['SeasonID'].unique()
    years.sort()

    for year in years:
        train_idxs = df.index[df['SeasonID'] != year]
        test_idxs = df.index[df['SeasonID'] == year]
        folds.append([train_idxs, test_idxs, year])

    return folds


def generate_outer_chronological_year_folds(df):
    folds = []

    # get years for iteration purposes
    years = df['SeasonID'].unique()
    years.sort()

    for year in years[2:]:
        train_idxs = df.index[df['SeasonID'] < year]
        test_idxs = df.index[df['SeasonID'] == year]
        folds.append([train_idxs, test_idxs, year])

    return folds


def generate_inner_chronological_year_folds(df):
    folds = []

    # get years for iteration purposes
    years = df['SeasonID'].unique()
    years.sort()

    train_idxs = df.index[df['SeasonID'] < years[-1]]
    test_idxs = df.index[df['SeasonID'] == years[-1]]
    folds.append([train_idxs, test_idxs, years[-1]])

    return folds


def generate_shuffled_folds(df):
    folds = []

    # # create KFold object - will be used for one of the fold strategies
    kf = KFold(
        n_splits=len(df['year'].unique()))

    fold = 1
    for train_idxs, test_idxs in kf.split(df):
        folds.append([train_idxs, test_idxs, fold])
        fold += 1

    return folds
