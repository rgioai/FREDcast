import itertools as it


def generate_csv():
    indicators = ['cpi', 'gdp', 'unemployment', 'payroll']
    norms = ['normal_dist', 'percent_change', 'zero_one']
    residuals = ['exp_residual', 'gdp_residual', 'linear_residual', 'none']
    test_number = ['test1', 'test2']

    order = list(it.product(indicators, norms, residuals, test_number))

    with open('tpot_results.csv', 'w+') as csvfile:
        cols = ['indicator', 'norm', 'residual', 'test_number', 'train_r2', 'all_r2', 'diff_r2', 'train_mse',
                'test_mse', 'all_mse', 'predicted', 'actual']

        for col in cols:
            csvfile.write(str(col) + ",")
        csvfile.write('\n')

        for i in range(0, len(order), 1):
            train_r2 = 'none'
            all_r2 = 'none'
            diff_r2 = 'none'
            train_mse = 'none'
            test_mse = 'none'
            all_mse = 'none'
            predicted = 'none'
            actual = 'none'

            path = 'tpot_results/{0}/{1}/{2}/{3}.out'.format(str(order[i][0]), str(order[i][1]), str(
                order[i][2]), str(order[i][3]))
            with open(path) as f:
                for line in f:
                    if 'train_r2:' in line.strip():
                        train_r2 = line.strip().split(':')[1]
                    elif 'all_r2:' in line.strip():
                        all_r2 = line.strip().split(':')[1]
                    elif 'diff_r2:' in line.strip():
                        diff_r2 = line.strip().split(':')[1]
                    elif 'train_mse:' in line.strip():
                        train_mse = line.strip().split(':')[1]
                    elif 'test_mse:' in line.strip():
                        test_mse = line.strip().split(':')[1]
                    elif 'all_mse:' in line.strip():
                        all_mse = line.strip().split(':')[1]
                    elif 'predicted:' in line.strip():
                        predicted = line.strip().split(':')[1]
                    elif 'actual:' in line.strip():
                        actual = line.strip().split(':')[1]

            if any(var is 'none' for var in
                   [train_r2, all_r2, diff_r2, train_mse, test_mse, all_mse, predicted, actual]):
                raise ValueError(str(path) + " is missing one of the stat variables.")

            else:
                csvfile.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}".format(str(order[i][0]),
                                                                                         str(order[i][1]),
                                                                                         str(order[i][2]),
                                                                                         str(order[i][3]),
                                                                                         str(train_r2),
                                                                                         str(all_r2),
                                                                                         str(diff_r2),
                                                                                         str(train_mse),
                                                                                         str(test_mse),
                                                                                         str(all_mse),
                                                                                         str(predicted),
                                                                                         str(actual)))
                csvfile.write('\n')


if __name__ == '__main__':
    generate_csv()
