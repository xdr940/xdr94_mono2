# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from my_trainer import Trainer
from opts.train_opts import train_opts
from opts.mc_train_opts import mc_train_opts



if __name__ == "__main__":
    options = train_opts()
    opts = options.args()
    trainer = Trainer(opts)
    trainer.train()
    print('training over')
