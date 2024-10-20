def batch_iterator(dataframe, batch_size=512):
    total_rows = dataframe.shape[0]
    for start_idx in range(0, total_rows, batch_size):
        yield dataframe.iloc[start_idx:start_idx + batch_size]