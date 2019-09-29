import sim_run_funcs

SIM_TYPE_FED_LEARNING = "fed_learning"
SIM_TYPE_FED_AVG_WITH_DP = "fed_avg_with_dp"

DATA_GEN_TYPE_DATA_FROM_FILE = "file_data"
DATA_GEN_TYPE_BLOB = "data_gen_blob"
DATA_GEN_TYPE_RAND = "data_gen_rand"

P_KEY_DATA_FILE_PATH = "data_file_path"

P_KEY_NUM_SAMPLES = "num_samples"
P_KEY_NUM_LABELS = "num_labels"
P_KEY_NUM_FEATURES = "num_features"
P_KEY_NUM_USERS = "num_users"

P_KEY_NUM_ROUNDS = "num_rounds"
P_KEY_BATCH_SIZE = "batch_size"
P_KEY_NUM_EPOCHS = "num_epochs"


class RunFuncAndReqParams:
    def __init__(self, run_func, prereq_check_func):
        self.run_func = run_func
        self.prereq_params = prereq_check_func


run_func_ltable = {
    SIM_TYPE_FED_LEARNING: RunFuncAndReqParams(
        sim_run_funcs.run_fed_learn_sim,
        {
            P_KEY_NUM_ROUNDS,
            P_KEY_BATCH_SIZE,
            P_KEY_NUM_EPOCHS,
            P_KEY_NUM_SAMPLES,
            P_KEY_NUM_FEATURES,
            P_KEY_NUM_USERS,
        },
    ),
    SIM_TYPE_FED_AVG_WITH_DP: RunFuncAndReqParams(
        sim_run_funcs.run_fed_avg_with_dp, {P_KEY_NUM_LABELS, P_KEY_NUM_FEATURES}
    ),
    DATA_GEN_TYPE_DATA_FROM_FILE: RunFuncAndReqParams(
        sim_run_funcs.read_data_from_file,
        {
            P_KEY_NUM_SAMPLES,
            P_KEY_NUM_LABELS,
            P_KEY_NUM_FEATURES,
            P_KEY_NUM_USERS,
            P_KEY_DATA_FILE_PATH,
        },
    ),
    DATA_GEN_TYPE_BLOB: RunFuncAndReqParams(
        sim_run_funcs.run_data_gen_blob,
        {P_KEY_NUM_SAMPLES, P_KEY_NUM_LABELS, P_KEY_NUM_FEATURES, P_KEY_NUM_USERS},
    ),
    DATA_GEN_TYPE_RAND: RunFuncAndReqParams(
        sim_run_funcs.run_data_gen_rand,
        {P_KEY_NUM_SAMPLES, P_KEY_NUM_LABELS, P_KEY_NUM_FEATURES, P_KEY_NUM_USERS},
    ),
}


class Simulation:
    def __init__(self):
        self._params = {}

    def run(self, sim_type, data_gen_type):
        sim_run_info = run_func_ltable[sim_type]
        data_gen_run_info = run_func_ltable[data_gen_type]

        if not self._sim_has_required_params_for_given_run_func(
            sim_run_info.prereq_params, sim_type
        ):
            return

        if not self._sim_has_required_params_for_given_run_func(
            sim_run_info.prereq_params, data_gen_type
        ):
            return

        print('Generating "{}" data...'.format(data_gen_type))
        generated_data = data_gen_run_info.run_func(self._params)

        print('Running the "{}" simulation...'.format(sim_type))
        sim_run_info.run_func(self._params, generated_data)

        print("Finished!")

    def set_num_samples(self, num_samples):
        return self._set_param(P_KEY_NUM_SAMPLES, num_samples)

    def set_num_labels(self, num_labels):
        return self._set_param(P_KEY_NUM_LABELS, num_labels)

    def set_num_features(self, num_features):
        return self._set_param(P_KEY_NUM_FEATURES, num_features)

    def set_num_users(self, num_users):
        return self._set_param(P_KEY_NUM_USERS, num_users)

    def set_read_data_file_path(self, file_path):
        return self._set_param(P_KEY_DATA_FILE_PATH, file_path)

    def set_num_rounds(self, num_rounds):
        return self._set_param(P_KEY_NUM_ROUNDS, num_rounds)

    def set_batch_size(self, batch_size):
        return self._set_param(P_KEY_BATCH_SIZE, batch_size)

    def set_num_epochs(self, num_epochs):
        return self._set_param(P_KEY_NUM_EPOCHS, num_epochs)

    def _set_param(self, p_key, val):
        self._params[p_key] = val
        return self

    def _sim_has_required_params_for_given_run_func(
        self, run_func_params, run_func_key
    ):
        missing_req_params = run_func_params.difference(self._params)

        if len(missing_req_params) != 0:
            print(
                "Can not run {} because the following required parameters are missing: ".format(
                    run_func_key
                )
            )

            for missing_p in missing_req_params:
                print("- {}".format(missing_p))

            return False
        return True
