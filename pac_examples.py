from models import conjunction, decision_list
from data import project_data

conjunction = conjunction.get_conjunction(project_data.train_data)
decision_list = decision_list.get_decision_list(project_data.train_data)



