from models import conjunction, decision_list
from data import project_data

conj_m = conjunction.get_approx_sample_size(epsilon=0.1, delta=0.1, n=85)
conj_train, conj_train_labels, conj_test, conj_test_labels = project_data.get_data_sample(conj_m)

dl_m = decision_list.get_approx_sample_size(epsilon=0.1, delta=0.1, n=85)
dl_train, dl_train_labels, dl_test, dl_test_labels = project_data.get_data_sample(dl_m)

h_conj = conjunction.get_conjunction(conj_train, conj_train_labels)
h_dl = decision_list.get_decision_list(dl_train, dl_train_labels)
