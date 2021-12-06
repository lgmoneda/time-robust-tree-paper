import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

def reverse_learning_curve_by_sample(train, holdout, model, features, target, time_column, time_sort, performance_function, n_rounds=5):
    results = {"round": [], "holdout_performance": [], "sample_size": []}
    initial_size = int(len(train) / n_rounds)
    sample_sizes = np.linspace(initial_size, len(train), n_rounds, dtype=int)

    for sample_size in sample_sizes:

        model.fit(train.sort_values(time_sort, ascending=False).iloc[:sample_size][features + [time_column]].reset_index(inplace=False, drop=True),
                  train.sort_values(time_sort, ascending=False).iloc[:sample_size][target].reset_index(inplace=False, drop=True))
        results["sample_size"].append(sample_size)
        performance = performance_function(holdout[target],
                                           model.predict_proba(holdout[features])[:, 1])
        results["holdout_performance"].append(performance)

    return results

def reverse_learning_curve(train, holdout, model, features, target, time_column,
                           performance_function, n_rounds=5, challenger=False, verbose=False, use_permutation_importance=True, dummy_time_column=None):
    train_time_segments = np.sort(train[time_column].unique())[::-1]

    results = {"round": [], "holdout_performance": [], "feature_importance": [],
               "sample_size": [], "last_period_included": [], "holdout_performance_by_period": []}
    initial_size = int(len(train) / n_rounds)
    sample_sizes = np.linspace(initial_size, len(train), n_rounds, dtype=int)
    
    if use_permutation_importance:
        sample_holdout = holdout.sample(frac=0.1)
    for nth, time_segment in enumerate(train_time_segments):
        if verbose: print(train_time_segments[:nth+1])
        if challenger:
            model.fit(train[train[time_column].isin(train_time_segments[:nth+1])][features + [time_column]].reset_index(inplace=False, drop=True),
            train[train[time_column].isin(train_time_segments[:nth+1])][target].reset_index(inplace=False, drop=True))
        else:
            if dummy_time_column:
                model.fit(train[train[time_column].isin(train_time_segments[:nth+1])][features + [dummy_time_column]].reset_index(inplace=False, drop=True), train[train[time_column].isin(train_time_segments[:nth+1])][target].reset_index(inplace=False, drop=True))
            else:
                model.fit(train[train[time_column].isin(train_time_segments[:nth+1])][features].reset_index(inplace=False, drop=True),
                train[train[time_column].isin(train_time_segments[:nth+1])][target].reset_index(inplace=False, drop=True))

        sample_size = train[time_column].isin(train_time_segments[:nth+1]).sum()
        ### Feature Importance
        if use_permutation_importance:

            model_importances = permutation_importance(model, sample_holdout[features], 
                                                       sample_holdout[target],
                                                       random_state=42, scoring=make_scorer(performance_function))
            model_importances = model_importances["importances_mean"]
            model_importances = pd.Series(model_importances, index=features)
        else:
            if trt_model:
                model_importances = model.feature_importance()

            else:
                model_importances = model.feature_importances_
                model_importances = pd.Series(model_importances, index=features)

        model_importances.rename(time_segment, inplace=True)
        results["feature_importance"].append(model_importances)

        results["sample_size"].append(sample_size)
        results["last_period_included"].append(time_segment)
        holdout["pred"] = model.predict_proba(holdout[features])[:, 1]
        performance = performance_function(holdout[target],
                                           holdout["pred"] )
        results["holdout_performance"].append(performance)
        results["holdout_performance_by_period"].append(holdout.groupby(time_column).apply(lambda x: performance_function(x[target], x["pred"])))

    return results

def plot_shap_difference(importance, mode="rank", rotate=False, title=None, cmap=None, threshold=0, save_as=None):
    """
    Compares shap importances between multiple groups.
    Function created by @tatasz.
    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the importances, with features as index, and each column corresponding to a specific group.
    mode : string
        If "rank", the plot will use ranks, else actual shap importances
    rotate : boolean
        Rotates x axis labels
    title : str
        Title of the plot to be displayed
    cmap : cmap
        A cmap to be used by plot
    """
    from matplotlib.pyplot import get_cmap

    if cmap is None:
        cmap = get_cmap('viridis')

    fig, ax1 = plt.subplots(figsize=(3,6), dpi=150)
    ax2 = ax1.twinx()

    df = importance
    df = df[df.max(axis=1) > threshold]
    if mode == 'rank':
        df = df.rank(axis=0, ascending=False, method='first')
        df = df.max().max() - df
    df = df.sort_values(by=df.columns[0])

    ids = pd.DataFrame({"group": df.index}).reset_index().groupby("group").transform("count").rank(method='first')
    ids = ids.apply(lambda x: cmap((x["index"] - 1) / (len(ids["index"].unique())-1)), axis=1)

    rot = 0;
    if rotate:
        rot = 90;

    df.transpose().plot(legend=False, rot=rot,
                        ax=ax1, color=ids, linewidth=10, alpha=0.5);

    if mode == 'rank':
        ax1.yaxis.set_ticks(df[df.columns[0]]);
        ax1.set_yticklabels(df.index);

        df = df.sort_values(by=df.columns[-1])
        ax2.set_ybound(ax1.get_ybound());
        ax2.yaxis.set_ticks(df[df.columns[-1]]);
        ax2.set_yticklabels(df.index);
    else:
        bound = ax1.get_ybound()
        mindist = (bound[1] - bound[0]) * 0.01

        df = df.sort_values(by=df.columns[0])
        diff = abs(df.diff()).fillna(1)
        df_left = df[diff[df.columns[0]] > mindist]
        ax1.yaxis.set_ticks(df_left[df_left.columns[0]]);
        ax1.set_yticklabels(df_left.index);

        df = df.sort_values(by=df.columns[-1])
        diff = abs(df.diff()).fillna(1)
        df_right = df[diff[df.columns[-1]] > mindist]
        ax2.set_ybound(bound);
        ax2.yaxis.set_ticks(df_right[df_right.columns[-1]]);
        ax2.set_yticklabels(df_right.index);

    for c in df.columns:
        plt.axvline(x=c, color='lightgrey', linestyle='--');

    if title is None:
        plt.title("Importance by group")
    else:
        plt.title(title)        
        
    if save_as is not None:
        ax1.set_rasterized(True)
        ax2.set_rasterized(True)
        fig.savefig(save_as + ".jpg", quality=95, bbox_inches = "tight")
        fig.savefig(save_as, format="eps", bbox_inches = "tight")

def plot_feature_migration_from_learning_curve_results(results, features, save_as=None, rotate=False):
    all_times_importance = pd.DataFrame(index=features)
    for imp in results["feature_importance"]:
        all_times_importance = all_times_importance.merge(imp, how="left", left_index=True,
                                right_index=True)

    all_times_importance = all_times_importance.applymap(np.abs)
    plot_shap_difference(all_times_importance, title="", save_as=save_as, rotate=rotate)
    
    return all_times_importance
