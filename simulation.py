SIM_TYPE_FED_LEARNING = 0

DATA_GEN_TYPE_REAL_DATA = 0
DATA_GEN_TYPE_BLOB = 1
DATA_GEN_TYPE_RAND = 2

P_KEY_NUM_SAMPLES = "num_samples"
P_KEY_NUM_LABELS = "num_labels"
P_KEY_NUM_FEATURES = "num_features"
P_KEY_NUM_USERS = "num_users"


def run_data_gen_blob(sim_params):
    pass


class RunFuncAndReqParams:
    def __init__(self, run_func, prereq_check_func):
        self.run_func = run_func
        self.prereq_check_func = prereq_check_func


run_func_ltable = {
    DATA_GEN_TYPE_BLOB: RunFuncAndReqParams(
        run_data_gen_blob,
        {P_KEY_NUM_SAMPLES, P_KEY_NUM_LABELS, P_KEY_NUM_FEATURES, P_KEY_NUM_USERS},
    ),
    DATA_GEN_TYPE_RAND: RunFuncAndReqParams(
        run_data_gen_blob,
        {P_KEY_NUM_SAMPLES, P_KEY_NUM_LABELS, P_KEY_NUM_FEATURES, P_KEY_NUM_USERS},
    ),
}


class Simulation:
    def __init__(self):
        self.params = {}

    def run(self, sim_type, data_gen_type):
        sim_run_info = run_func_ltable[sim_type]
        data_gen_run_info = run_func_ltable[data_gen_type]

        if not self._sim_has_required_params_for_given_run_func(sim_type):
            return

        if not self._sim_has_required_params_for_given_run_func(data_gen_type):
            return

        print('Generating "{}" data...'.format(data_gen_type))
        generated_data = data_gen_run_info.run_func(self.params)

        print('Runing the "{}" simulation...'.format(sim_type))
        sim_run_info.run_func(self.params, generated_data)

        print("Finished!")

    def set_num_samples(self, num_samples):
        return self._set_param(P_KEY_NUM_SAMPLES, num_samples)

    def set_num_labels(self, num_labels):
        return self._set_param(P_KEY_NUM_LABELS, num_labels)

    def set_num_features(self, num_features):
        return self._set_param(P_KEY_NUM_FEATURES, num_features)

    def set_num_users(self, num_users):
        return self._set_param(P_KEY_NUM_USERS, num_users)

    def _set_param(self, p_key, val):
        self.param[p_key] = val
        return self

    def _sim_has_required_params_for_given_run_func(
        self, run_func_params, run_func_key
    ):
        missing_req_params = run_func_params.difference(self.params)

        if len(missing_req_params) != 0:
            print(
                "Can not run {} because the following required parameters are missing: ".format(
                    run_func_key
                )
            )

            for missing_p in missing_req_params:
                print(missing_p)

            return False
        return True
