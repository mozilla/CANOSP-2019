import runner_run_funcs
import json


class RunnerException(Exception):
    def __init__(self, msg):
        super(RunnerException, self).__init__(msg)


class Runner:

    run_func_ltable, data_gen_func_ltable = runner_run_funcs.get_run_funcs()

    def __init__(self, param_json, sim_type, data_gen_type):
        self._params = json.loads(param_json)

        sim_run_func, sim_prereq_params = Runner.run_func_ltable[sim_type]
        gen_run_func, gen_prereq_params = Runner.data_gen_func_ltable[data_gen_type]

        self._verify_sim_has_required_params_for_given_run_func(
            sim_prereq_params, sim_type
        )
        self._verify_sim_has_required_params_for_given_run_func(
            gen_prereq_params, data_gen_type
        )

        print('Generating "{}" data...'.format(data_gen_type))
        generated_data = gen_run_func(self._params)

        print('Running the "{}" simulation...'.format(sim_type))
        sim_run_func(self._params, generated_data)

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
