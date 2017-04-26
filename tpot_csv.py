import itertools as it


def generate_csv():
    indicators = ['cpi', 'gdp', 'unemployment', 'payroll']
    norms = ['normal_dist', 'percent_change', 'zero_one']
    residuals = ['exp_residual', 'gdp_residual', 'linear_residual', 'none']
    test_number = ['test1', 'test2']
    train_r2 = 0.998577726596
    all_r2 = 0.998538055207
    diff_r2 = 3.96713893445e-05
    train_mse = 1.68695903106
    test_mse = 8.86559735703
    all_mse = 1.75261730843
    predicted = 241.027700806
    actual = 243.752

    order = list(it.product(indicators, norms, residuals, test_number))

    with open('tpot_results.csv', 'w+') as csvfile:
        cols = ['indicator', 'norm', 'residual', 'test_number', 'train_r2', 'all_r2', 'diff_r2', 'train_mse',
                'test_mse', 'all_mse', 'predicted', 'actual']

        for col in cols:
            csvfile.write(str(col) + ",")
        csvfile.write('\n')

        for i in range(0, len(order), 1):
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
