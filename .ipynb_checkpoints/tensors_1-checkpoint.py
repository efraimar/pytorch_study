import torch
import numpy as np

# make tensor
data = [[3,1],[4,2]]
x = torch.Tensor(data)
print(x)

# make from numpy
np_array = np.array(data)

# טנסור ממערך
x = torch.tensor([1, 2, 3])

# טנסור מלא באפסים
zeros = torch.zeros(2, 3)


# טנסור מלא באחדות
ones = torch.ones(4, 2)

# טנסור עם ערכים אקראיים
random_tensor = torch.rand(3,3)
print(random_tensor)
