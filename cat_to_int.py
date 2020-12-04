def cat_to_int(feature, target, return_dict=True):
    '''
    Convert a categorical feature to integer values according to its relationship with a binary target.
    The feature is sorted by Weight of Evidence in ascending order and transformed to nonnegative integers.
            Parameters
            ----------
                    feature : array_like
                        Input array or object that can be converted to an array. Each element in the array is
                        an observed value of an independent variable used to model `target`.
                    target : array_like
                        Input array or object that can be converted to an array. Each element in the array is
                        the observed outcome of an experiment, where 0 means non-event and 1 means event.
                    return_dict : bool, default `True`
                        If `return_dict == True`, the function returns a dictionary to map the levels of `feature`
                        to nonnegative integers.
                        If `return_dict == False`, the function returns a `pandas.DataFrame` with all the inputs
                        used to calculate the Weight of Evidence.
            Returns
            -------
                    Python dictionary or `pandas.DataFrame` (depending on the value of `returnd_dict`).
            Notes
            -----
                    `feature` is coerced to `str`. Thus, all types of null values are translated to text, such as
                    'nan', 'NaN', etc.
            GitHub
            ------
                    Visit https://github.com/ArturoSbr for more content.
    '''
    t = pd.DataFrame({'level':feature, 'y':target})
    t['level'] = t['level'].astype('str')
    ev = t['y'].sum()
    nv = len(t) - ev
    t = t.groupby('level')['y'].sum().reset_index(name='events')
    t['non_events'] = ((ev + nv) - t['events']).div(nv)
    t['events'] = t['events'].div(ev)
    t['woe'] = np.log(t['non_events'].replace(0, 0.01) / t['events'].replace(0, 0.01))
    t = t.sort_values('woe', ascending=True)
    t.reset_index(drop=True, inplace=True)
    if return_dict:
        return {v:i for i, v in t['level'].to_dict().items()}
    else:
        return t.assign(integer_value=range(len(t)))