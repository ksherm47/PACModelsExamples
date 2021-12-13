from data import project_data
from experiments import conjunction_experiment, disjunction_experiment, three_cnf_experiment, decision_list_experiment
import matplotlib.pyplot as plt

RUN_CONJUNCTION_EXPERIMENT = False
RUN_DISJUNCTION_EXPERIMENT = False
CONJUNCTION_TRIALS = 1000
DISJUNCTION_TRIALS = 1000
CONJUNCTION_EPSILON = 0.1  # Both epsilon & delta used for disjunction learning too
CONJUNCTION_DELTA = 0.1

RUN_DECISION_LIST_EXPERIMENT = False
DECISION_LIST_TRIALS = 1000
DECISION_LIST_EPSILON = 0.15
DECISION_LIST_DELTA = 0.1

RUN_3CNF_EXPERIMENT = False

UNZIP_DATA_OBJECTS = False
DATA_OBJECTS_TO_ZIP = ['3CNF_hypothesis']
ZIP_DATA_OBJECTS = False


if UNZIP_DATA_OBJECTS:
    project_data.unzip_data()

if RUN_CONJUNCTION_EXPERIMENT:
    conj_errs = conjunction_experiment(epsilon=CONJUNCTION_EPSILON,
                                       delta=CONJUNCTION_DELTA,
                                       num_trials=CONJUNCTION_TRIALS,
                                       verbose=False,
                                       improved_sample_size=True)
    delta_rate = sum([1 for err in conj_errs if err < CONJUNCTION_EPSILON]) / CONJUNCTION_TRIALS

    print(f'Rate that conjunction had error less than {CONJUNCTION_EPSILON}: {delta_rate}')

    plt.hist(conj_errs, bins=20)
    plt.title(f'Conjunction error rates ({CONJUNCTION_TRIALS} trials, epsilon={CONJUNCTION_EPSILON}, delta={CONJUNCTION_DELTA})')
    plt.show()

if RUN_DISJUNCTION_EXPERIMENT:
    disj_errs = disjunction_experiment(epsilon=CONJUNCTION_EPSILON,
                                       delta=CONJUNCTION_DELTA,
                                       num_trials=DISJUNCTION_TRIALS,
                                       verbose=False)
    delta_rate = sum([1 for err in disj_errs if err < CONJUNCTION_EPSILON]) / DISJUNCTION_TRIALS

    print(f'Rate that disjunction had error less than {CONJUNCTION_EPSILON}: {delta_rate}')

    plt.hist(disj_errs, bins=20)
    plt.title(f'Disjunction error rates ({DISJUNCTION_TRIALS} trials, epsilon={CONJUNCTION_EPSILON}, delta={CONJUNCTION_DELTA})')
    plt.show()

if RUN_DECISION_LIST_EXPERIMENT:
    dl_errs = decision_list_experiment(epsilon=DECISION_LIST_EPSILON,
                                       delta=DECISION_LIST_DELTA,
                                       num_trials=DECISION_LIST_TRIALS,
                                       verbose=False)
    delta_rate = sum([1 for err in dl_errs if err < DECISION_LIST_EPSILON]) / DECISION_LIST_TRIALS

    print(f'Rate that decision list had error less than {DECISION_LIST_EPSILON}: {delta_rate}')
    plt.hist(dl_errs, bins=20)
    plt.title(f'Decision List error rates ({DECISION_LIST_TRIALS} trials, epsilon={DECISION_LIST_EPSILON}, delta={DECISION_LIST_DELTA})')
    plt.show()

if RUN_3CNF_EXPERIMENT:
    three_cnf_error = three_cnf_experiment()

if ZIP_DATA_OBJECTS:
    project_data.zip_data(DATA_OBJECTS_TO_ZIP, clean=True, remove_previous=True)
