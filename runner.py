import runner_run_funcs
import json


class RunnerException(Exception):
    def __init__(self, msg):
        super(RunnerException, self).__init__(msg)


class Runner:

    run_func_ltable, data_gen_func_ltable = runner_run_funcs.get_run_funcs()

    def __init__(self, param_json, s_type, d_type):
        self._params = json.loads(param_json)

        s_run_func, s_prq_prms = Runner.run_func_ltable[s_type]
        g_run_func, g_prq_prms = Runner.data_gen_func_ltable[d_type]

        self._verify_sim_has_required_params_for_given_run_func(s_prq_prms, s_type)
        self._verify_sim_has_required_params_for_given_run_func(g_prq_prms, d_type)

        print('Generating "{}" data...'.format(d_type))
        generated_data = g_run_func(self._params)

        print('Running the "{}" simulation...'.format(s_type))
        s_run_func(self._params, generated_data)

        print("Finished!")

    def _verify_sim_has_required_params_for_given_run_func(
        self, run_func_params, run_func_key
    ):
        missing_req_params = run_func_params.difference(self._params)

        if len(missing_req_params) == 0:
            return

        ex_msg = "Can not run {} because the following required parameters are missing: ".format(
            run_func_key
        )

        for missing_p in missing_req_params:
            ex_msg += "\n- {}".format(missing_p)

        raise RunnerException(ex_msg)
