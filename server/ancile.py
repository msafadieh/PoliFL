from ancile.core.core import execute

def execute_program(dpps, status_queue, rpc_queue):
    res = execute(users_secrets=[],
            program=program,
            data_policy_pairs=dpps,
            app_id=None,
            app_module=None,
            rpc_queue=rpc_queue)
    status_queue.put_nowait(res)

