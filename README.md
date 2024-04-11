# AirGNN-Reproducability
 
This repo contains the implementation of AirGNN introduced in "Graph Neural Networks with Adaptive Residual" by Liu et.al. We describe the use case of each of the filed in the following lines.

train_model.py: trains the specified model on the specified dataset. Model can be either "AirGNN", "APPNP", "GCN", "GCNII", "GAT", or "MLP".

adv_attack.py: produces the adversarial data.

test_adv.py: can be used to test the trained model in an adversarial setting where adversarial data are produced usign "adv_attack.py".

noise_test.py: Can be used to test the trained model in a noisy setting where the noises are drawn from a Multivariate Gaussian Distribution.

residual_test.py: Checks the residual score of each node in the noisy scenario.

residual_adv_test.py: Checks the residual score of each node in the adversarial setting.

Results of the runs can be found in adv/fixed_data/model/noisy directories.
