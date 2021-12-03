from models import conjunction, decision_list
from data import project_data


def conjunction_experiment(num_trials=1000, epsilon=0.1, delta=0.1, m=None):
    if m is None:
        conj_m = conjunction.get_approx_sample_size(epsilon=epsilon, delta=delta, n=project_data.data_dim)
    else:
        conj_m = m
    print(f'Conducting Conjunction experiment with sample size {m} and {num_trials} trials.')

    error_rates = []
    for _ in range(num_trials):
        conj_train, conj_train_labels, conj_test, conj_test_labels = project_data.get_data_sample(conj_m)
        h_conj = conjunction.get_conjunction(conj_train, conj_train_labels)

        # Evaluating on train and test data for population error rate
        error_rate = 0
        for test_data_point, test_label in zip(conj_test, conj_test_labels):
            error_rate += 1 if test_label != h_conj.evaluate(test_data_point) else 0
        for train_data_point, train_label in zip(conj_train, conj_train_labels):
            error_rate += 1 if train_label != h_conj.evaluate(train_data_point) else 0
        error_rate /= (conj_test.shape[0] + conj_train.shape[0])

        print(h_conj)
        print(f'Error Rate: {error_rate}')

        error_rates.append(error_rate)


def decision_list_experiment(num_trials=1000, epsilon=0.4, delta=0.1, m=None):
    if m is None:
        dl_m = decision_list.get_approx_sample_size(epsilon=epsilon, delta=delta, n=project_data.data_dim)
    else:
        dl_m = m
    print(f'Conducting Decision List experiment with sample size {m} and {num_trials} trials.')

    error_rates = []
    for _ in range(num_trials):
        dl_train, dl_train_labels, dl_test, dl_test_labels = project_data.get_data_sample(dl_m)
        h_dl = decision_list.get_decision_list(dl_train, dl_train_labels)

        error_rate = 0
        for test_data_point, test_label in zip(dl_test, dl_test_labels):
            error_rate += 1 if test_label != h_dl.evaluate(test_data_point) else 0
        for train_data_point, train_label in zip(dl_train, dl_train_labels):
            error_rate += 1 if train_label != h_dl.evaluate(train_data_point) else 0
        error_rate /= (dl_test.shape[0] + dl_train.shape[0])

        print(h_dl)
        print(f'Error Rate:{error_rate}')

        error_rates.append(error_rate)


conjunction_experiment(num_trials=10)
decision_list_experiment(num_trials=10)
