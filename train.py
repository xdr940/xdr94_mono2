# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from my_trainer import Trainer
from options import MD_train_opts,MC_train_options




if __name__ == "__main__":
    options = MD_train_opts()
    opts = options.parse()
    trainer = Trainer(opts)
    trainer.train()
    print('training over')
