def split_data(data, train_size=0.7):
    train_size = int(train_size*data.shape[1])
    train_data = data[:, :train_size, :]
    test_data = data[:, train_size:, :]
    return train_data, test_data