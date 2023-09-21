# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from models.QCNet.losses.focal_loss import FocalLoss
from models.QCNet.losses.gaussian_nll_loss import GaussianNLLLoss
from models.QCNet.losses.laplace_nll_loss import LaplaceNLLLoss
from models.QCNet.losses.mixture_nll_loss import MixtureNLLLoss
from models.QCNet.losses.mixture_of_gaussian_nll_loss import MixtureOfGaussianNLLLoss
from models.QCNet.losses.mixture_of_laplace_nll_loss import MixtureOfLaplaceNLLLoss
from models.QCNet.losses.mixture_of_von_mises_nll_loss import MixtureOfVonMisesNLLLoss
from models.QCNet.losses.nll_loss import NLLLoss
from models.QCNet.losses.soft_target_cross_entropy_loss import SoftTargetCrossEntropyLoss
from models.QCNet.losses.von_mises_nll_loss import VonMisesNLLLoss
