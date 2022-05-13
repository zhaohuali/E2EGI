
import os

from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB


def set_superparameters(args, param_config):

    for key, value in param_config.items():
        if key in vars(args):
            args.__dict__[key] = value
        else:
            raise ValueError(f'key: {key} not in args')


def ray_tune_main(main, args):

    ray_root = os.path.join(args.root, 'history/ray_tune')

    _default_config = {
        'epochs': tune.randint(5000, 20_000),
        'lr': tune.uniform(1e-1, 1),
        'TV': tune.loguniform(1e-6, 0.1),
        'BN': tune.loguniform(1e-6, 0.1),
        'input_norm': tune.uniform(-1e-6, 0),
        'grads_weight': tune.uniform(1e-2, 1),
        'verbose': args.verbose,
    }

    config = dict()
    if args.epochs_tune:
        config['epochs'] = _default_config['epochs']
    if args.lr_tune:
        config['lr'] = _default_config['lr']
    if args.TV_tune:
        config['TV'] = _default_config['TV']
    if args.BN_tune:
        config['BN'] = _default_config['BN']
    if args.input_norm_tune:
        config['input_norm'] = _default_config['input_norm']
    if args.grads_weight_tune:
        config['grads_weight'] = _default_config['grads_weight']

    print('ray tune config')
    for key, value in config.items():
        msg = f'tune [{key:30}] True'
        print(msg)

    bohb_hyperband = HyperBandForBOHB(
        max_t=args.max_t,
        reduction_factor=4,
        stop_last_trials=True, )
    bohb_search = TuneBOHB(seed=args.seed)
    bohb_search = tune.suggest.ConcurrencyLimiter(
        bohb_search, max_concurrent=args.max_concurrent)
    result = tune.run(main,
                      metric='loss',
                      mode='min',
                      config=config,
                      search_alg=bohb_search,
                      scheduler=bohb_hyperband,
                      num_samples=args.num_samples,
                      resources_per_trial={'gpu': args.ngpus_per_trial},
                      local_dir=ray_root,
                      verbose=args.verbose
                      )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    logdir = result.get_best_logdir("loss", mode="min")
    print("Best trial logdir: {}".format(logdir))

    return best_trial.config
