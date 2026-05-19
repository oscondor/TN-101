import mps_class 
import numpy as np
import mps_library as mps_lib
import mpo_class
import DMRG 

tensor = np.random.rand(10,10, 10,10,10,10)

eval_vector = [1,2,0,2,3,4]
mpo = mpo_class.MPO(tensor)
print([mpo.cores[i].shape for i in range(len(mpo.cores))])

mpo_add = mpo.addition(mpo)
print([mpo_add.cores[i].shape for i in range(len(mpo.cores))])
mps_add = mpo_add.mpo_to_mps()
print([mps_add.cores[i].shape for i in range(len(mps_add.cores))])
mps_add.sweep_left(None)
mps_add.sweep_right(None)
mpo_reconstructed = mpo_class.MPO.mps_to_mpo(mps_add)
print([mpo_reconstructed.cores[i].shape for i in range(len(mpo_reconstructed.cores))])