
import os

import torch
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from .gradient_inversion import BaseGradientInversion


class HSGradientInversion(BaseGradientInversion):

    def __init__(self, target_gradient, model, dm, ds, args) -> None:
        super().__init__(target_gradient, model, dm, ds, args)
        self.min_t = args.min_t

    def run(self, lr_start_end,
            regularization, x_pseudo,
            y_pseudo, epochs,
            metric_dict,
            bn_loss_layers=None, x_group=None,
            state='update'):

        '''
        lr_start_end: tuple (lr_start, lr_end)
        regularization: dict ('TV': value, .., )
        '''

        self.setup(regularization, bn_loss_layers, x_group)
        x_true = metric_dict['x_true'].cpu()

        x_pseudo = x_pseudo.detach().clone()
        x_pseudo.requires_grad = True

        if self.optim == 'Adam':
            optimizer = torch.optim.Adam([x_pseudo], lr_start_end[0])

        scheduler = self.get_lr_scheduler(optimizer, epochs, lr_start_end[1])

        try:
            for epoch in range(1, epochs+1):
                closure = self.grads_inversion(x_pseudo, y_pseudo,
                                               optimizer, state)
                optimizer.step(closure)
                scheduler.step()

                with torch.no_grad():

                    # add noise to x_pseudo
                    if self.input_noise > 0:
                        x_pseudo.data = self.add_noise(x_pseudo, optimizer)

                    # boxing x_pseudo
                    if self.input_boxed:
                        x_pseudo.data = self.boxing_input(x_pseudo)

                    if (epoch == 1
                            and epoch % self.print_freq == 0
                            and epoch == epochs):
                        self.display_results(epoch, epochs)

                    # superparameters search
                    if epoch >= self.min_t:
                        if self.min_grads_loss:
                            loss = self.get_HP_loss(
                                self.best_x_pseudo.cpu(), x_true)
                        else:
                            loss = self.get_HP_loss(x_pseudo.cpu(), x_true)
                        tune.report(
                            iterations=epoch,
                            loss=loss
                        )

        except KeyboardInterrupt:
            print('early stop.')

        finally:
            self.display_results(epoch, epochs)

        if self.min_grads_loss:
            return self.best_x_pseudo, self.best_loss_track
        else:
            return x_pseudo, self.loss_track

    def get_HP_loss(self, x_pseudo, x_true):
        image_diff = 0
        image_diff += ((x_pseudo - x_true) ** 2).mean()
        return image_diff.item()


def set_superparameters(args, param_config):

    for key, value in param_config.items():
        if key in vars(args):
            args.__dict__[key] = value
        else:
            raise ValueError(f'key: {key} not in args')


def ray_tune_main(main, args):

    ray_root = os.path.join(args.root, 'results/ray_tune')

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
