from models import conjunction, disjunction, decision_list, three_cnf
from data import project_data
import matplotlib.pyplot as plt

RUN_CONJUNCTION_EXPERIMENT = False
RUN_DISJUNCTION_EXPERIMENT = False
RUN_DECISION_LIST_EXPERIMENT = True
RUN_3CNF_EXPERIMENT = False
UNZIP_DATA_OBJECTS = False
DATA_OBJECTS_TO_ZIP = ['3CNF_hypothesis']
ZIP_DATA_OBJECTS = False


def conjunction_experiment(num_trials=1000, epsilon=0.1, delta=0.1,
                           m=None, improved_sample_size=True, verbose=True) -> list[float]:
    if m is None:
        conj_m = conjunction.get_approx_sample_size(epsilon=epsilon,
                                                    delta=delta,
                                                    n=project_data.data_dim,
                                                    improved=improved_sample_size)
    else:
        conj_m = m
    print(f'========== Conducting Conjunction experiment with sample size {conj_m} and {num_trials} trials. ==========')

    error_rates = []
    for trial in range(num_trials):
        conj_train, conj_train_labels, conj_test, conj_test_labels = project_data.get_data_sample(conj_m)
        h_conj = conjunction.get_conjunction(conj_train, conj_train_labels, verbose=verbose)

        # Evaluating on train and test data for population error rate
        error_rate = 0
        for test_data_point, test_label in zip(conj_test, conj_test_labels):
            error_rate += 1 if test_label != h_conj.evaluate(test_data_point) else 0
        for train_data_point, train_label in zip(conj_train, conj_train_labels):
            error_rate += 1 if train_label != h_conj.evaluate(train_data_point) else 0
        error_rate /= (conj_test.shape[0] + conj_train.shape[0])

        if verbose:
            print(f'Trial {trial + 1}:')
            print(f'\tConjunction Hypothesis: {h_conj}')
            print(f'\tError Rate: {error_rate}\n')
        else:
            print(f'\r{trial + 1}/{num_trials} trials completed', end='')

        error_rates.append(error_rate)

    if not verbose:
        print('\n', end='')

    return error_rates


def disjunction_experiment(num_trials=1000, epsilon=0.1, delta=0.1,
                           m=None, improved_sample_size=True, verbose=True) -> list[float]:
    if m is None:
        disj_m = conjunction.get_approx_sample_size(epsilon=epsilon,
                                                    delta=delta,
                                                    n=project_data.data_dim,
                                                    improved=improved_sample_size)
    else:
        disj_m = m
    print(f'========== Conducting Disjunction experiment with sample size {disj_m} and {num_trials} trials. ==========')

    error_rates = []
    for trial in range(num_trials):
        disj_train, disj_train_labels, disj_test, disj_test_labels = project_data.get_data_sample(disj_m)
        h_disj = disjunction.get_disjunction(disj_train, disj_train_labels, verbose=verbose)

        # Evaluating on train and test data for population error rate
        error_rate = 0
        for test_data_point, test_label in zip(disj_test, disj_test_labels):
            error_rate += 1 if test_label != h_disj.evaluate(test_data_point) else 0
        for train_data_point, train_label in zip(disj_train, disj_train_labels):
            error_rate += 1 if train_label != h_disj.evaluate(train_data_point) else 0
        error_rate /= (disj_test.shape[0] + disj_train.shape[0])

        if verbose:
            print(f'Trial {trial + 1}:')
            print(f'\tDisjunction Hypothesis: {h_disj}')
            print(f'\tError Rate: {error_rate}\n')
        else:
            print(f'\r{trial + 1}/{num_trials} trials completed', end='')

        error_rates.append(error_rate)

    if not verbose:
        print('\n', end='')

    return error_rates


def decision_list_experiment(num_trials=1000, epsilon=0.15, delta=0.1, m=None, verbose=True) -> list[float]:
    if m is None:
        dl_m = decision_list.get_approx_sample_size(epsilon=epsilon,
                                                    delta=delta,
                                                    n=project_data.data_dim)
    else:
        dl_m = m
    print(f'========== Conducting Decision List experiment with sample size {dl_m} and {num_trials} trials ==========')

    error_rates = []
    for trial in range(num_trials):
        dl_train, dl_train_labels, dl_test, dl_test_labels = project_data.get_data_sample(dl_m)
        h_dl = decision_list.get_decision_list(dl_train, dl_train_labels)

        error_rate = 0
        for test_data_point, test_label in zip(dl_test, dl_test_labels):
            error_rate += 1 if test_label != h_dl.evaluate(test_data_point) else 0
        for train_data_point, train_label in zip(dl_train, dl_train_labels):
            error_rate += 1 if train_label != h_dl.evaluate(train_data_point) else 0
        error_rate /= (dl_test.shape[0] + dl_train.shape[0])

        if verbose:
            print(f'Trial {trial + 1}:')
            print(f'\tDecision List Model: {h_dl}')
            print(f'\tError Rate:{error_rate}\n')
        else:
            print(f'\r{trial + 1}/{num_trials} trials completed', end='')

        error_rates.append(error_rate)

    if not verbose:
        print('\n', end='')

    return error_rates


def three_cnf_experiment() -> float:
    full_data, full_data_labels = project_data.get_full_data()
    h_3cnf = three_cnf.get_three_cnf(full_data, full_data_labels)

    print('========== Conducting 3CNF experiment with full data set ==========')

    error_rate = 0
    for data_point, label in zip(full_data, full_data_labels):
        error_rate += 1 if label == h_3cnf.evaluate(data_point) else 0
    error_rate /= full_data.shape[0]

    print(f'3CNF Model: {h_3cnf}')
    print(f'3CNF Size: {h_3cnf.size()}')
    print(f'3CNF Error Rate: {error_rate}')

    return error_rate


if UNZIP_DATA_OBJECTS:
    project_data.unzip_data()
if RUN_CONJUNCTION_EXPERIMENT:
    conj_errs = conjunction_experiment(verbose=False)
    plt.hist(conj_errs, bins=20)
    plt.show()
if RUN_DISJUNCTION_EXPERIMENT:
    disj_errs = disjunction_experiment(num_trials=10)
    plt.hist(disj_errs, bins=20)
    plt.show()
if RUN_DECISION_LIST_EXPERIMENT:
    dl_errs = decision_list_experiment(verbose=False)
    plt.hist(dl_errs, bins=20)
    plt.show()
if RUN_3CNF_EXPERIMENT:
    three_cnf_error = three_cnf_experiment()
if ZIP_DATA_OBJECTS:
    project_data.zip_data(DATA_OBJECTS_TO_ZIP, clean=True, remove_previous=True)


