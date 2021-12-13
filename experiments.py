from data import project_data
from models import conjunction, disjunction, decision_list, three_cnf


def conjunction_experiment(epsilon, delta, num_trials,
                           m=None, improved_sample_size=False, tolerant=False, mu=0.8,
                           verbose=False, only_test=True, save_models=True) -> list[float]:
    if m is None:
        conj_m = conjunction.get_approx_sample_size(epsilon=epsilon,
                                                    delta=delta,
                                                    n=project_data.data_dim,
                                                    improved=improved_sample_size,
                                                    tolerant=tolerant,
                                                    mu=mu)
    else:
        conj_m = m
    print(f'========== Conducting Conjunction experiment with sample size {conj_m} and {num_trials} trials. ==========')

    error_rates = []
    models = []
    for trial in range(num_trials):
        conj_train, conj_train_labels, conj_test, conj_test_labels = project_data.get_data_sample(conj_m)
        h_conj = conjunction.get_conjunction(conj_train, conj_train_labels, verbose=verbose)
        models.append(h_conj)

        # Evaluating on train and test data for population error rate
        error_rate = 0
        for test_data_point, test_label in zip(conj_test, conj_test_labels):
            error_rate += 1 if test_label != h_conj.evaluate(test_data_point) else 0
        if not only_test:
            for train_data_point, train_label in zip(conj_train, conj_train_labels):
                error_rate += 1 if train_label != h_conj.evaluate(train_data_point) else 0
        error_rate /= (conj_test.shape[0] + (conj_train.shape[0] if not only_test else 0))

        if verbose:
            print(f'Trial {trial + 1}:')
            print(f'\tConjunction Hypothesis: {h_conj}')
            print(f'\tError Rate: {error_rate}\n')
        else:
            print(f'\r{trial + 1}/{num_trials} trials completed', end='')

        error_rates.append(error_rate)

    if not verbose:
        print('\n', end='')

    if save_models:
        project_data.save_data_obj(models, 'conj_models')

    return error_rates


def disjunction_experiment(epsilon, delta, num_trials,
                           m=None, improved_sample_size=False, tolerant=False, mu=0.8,
                           verbose=False, only_test=True, save_models=True) -> list[float]:
    if m is None:
        disj_m = conjunction.get_approx_sample_size(epsilon=epsilon,
                                                    delta=delta,
                                                    n=project_data.data_dim,
                                                    improved=improved_sample_size,
                                                    tolerant=tolerant,
                                                    mu=mu)
    else:
        disj_m = m
    print(f'========== Conducting Disjunction experiment with sample size {disj_m} and {num_trials} trials. ==========')

    error_rates = []
    models = []
    for trial in range(num_trials):
        disj_train, disj_train_labels, disj_test, disj_test_labels = project_data.get_data_sample(disj_m)
        h_disj = disjunction.get_disjunction(disj_train, disj_train_labels, verbose=verbose)
        models.append(h_disj)

        # Evaluating on train and test data for population error rate
        error_rate = 0
        for test_data_point, test_label in zip(disj_test, disj_test_labels):
            error_rate += 1 if test_label != h_disj.evaluate(test_data_point) else 0
        if not only_test:
            for train_data_point, train_label in zip(disj_train, disj_train_labels):
                error_rate += 1 if train_label != h_disj.evaluate(train_data_point) else 0
        error_rate /= (disj_test.shape[0] + (disj_train.shape[0] if not only_test else 0))

        if verbose:
            print(f'Trial {trial + 1}:')
            print(f'\tDisjunction Hypothesis: {h_disj}')
            print(f'\tError Rate: {error_rate}\n')
        else:
            print(f'\r{trial + 1}/{num_trials} trials completed', end='')

        error_rates.append(error_rate)

    if not verbose:
        print('\n', end='')

    if save_models:
        project_data.save_data_obj(models, 'disj_models')

    return error_rates


def decision_list_experiment(epsilon, delta, num_trials, m=None,
                             verbose=False, only_test=True, save_models=True) -> list[float]:
    if m is None:
        dl_m = decision_list.get_approx_sample_size(epsilon=epsilon,
                                                    delta=delta,
                                                    n=project_data.data_dim)
    else:
        dl_m = m
    print(f'========== Conducting Decision List experiment with sample size {dl_m} and {num_trials} trials ==========')

    error_rates = []
    models = []
    for trial in range(num_trials):
        dl_train, dl_train_labels, dl_test, dl_test_labels = project_data.get_data_sample(dl_m)
        h_dl = decision_list.get_decision_list(dl_train, dl_train_labels)
        models.append(h_dl)

        error_rate = 0
        for test_data_point, test_label in zip(dl_test, dl_test_labels):
            error_rate += 1 if test_label != h_dl.evaluate(test_data_point) else 0
        if not only_test:
            for train_data_point, train_label in zip(dl_train, dl_train_labels):
                error_rate += 1 if train_label != h_dl.evaluate(train_data_point) else 0
        error_rate /= (dl_test.shape[0] + (dl_train.shape[0] if not only_test else 0))

        if verbose:
            print(f'Trial {trial + 1}:')
            print(f'\tDecision List Model: {h_dl}')
            print(f'\tError Rate:{error_rate}\n')
        else:
            print(f'\r{trial + 1}/{num_trials} trials completed', end='')

        error_rates.append(error_rate)

    if not verbose:
        print('\n', end='')

    if save_models:
        project_data.save_data_obj(models, 'dl_models')

    return error_rates


def three_cnf_experiment(use_full_data=True, m=project_data.total_data.shape[0],
                         only_test=True, hypothesis_name='3CNF_hypothesis', print_model=False) -> float:
    if use_full_data:
        three_cnf_train, three_cnf_train_labels = project_data.get_full_data()
        three_cnf_test = three_cnf_train
        three_cnf_test_labels = three_cnf_train_labels
        m = three_cnf_train.shape[0]
    else:
        three_cnf_train, three_cnf_train_labels, three_cnf_test, three_cnf_test_labels = project_data.get_data_sample(m)

    h_3cnf = three_cnf.get_three_cnf(three_cnf_train, three_cnf_train_labels, hypothesis_name=hypothesis_name)

    print(f'========== Conducting 3CNF experiment with sample size {m} ==========')

    error_rate = 0
    for test_data_point, test_label in zip(three_cnf_test, three_cnf_test_labels):
        error_rate += 1 if test_label == h_3cnf.evaluate(test_data_point) else 0
    if not use_full_data and not only_test:
        for train_data_point, train_label in zip(three_cnf_train, three_cnf_train_labels):
            error_rate += 1 if train_label == h_3cnf.evaluate(train_data_point) else 0
    error_rate /= (three_cnf_test.shape[0] + (three_cnf_train.shape[0] if not use_full_data and not only_test else 0))

    if print_model:
        print(f'3CNF Model: {h_3cnf}')
    print(f'3CNF Size: {h_3cnf.size()}')
    print(f'3CNF Error Rate: {error_rate}')

    return error_rate
