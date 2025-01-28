import torch
import numpy as np

# יצירת טנסור ממערך
data = [[3, 1], [4, 2]]
x = torch.Tensor(data)
print("Tensor from list:", x)

# יצירת טנסור מ-Numpy
np_array = np.array(data)
to_tensor = torch.from_numpy(np_array)
print("Tensor from numpy:", to_tensor)

# דוגמאות לטנסורים שונים
# טנסור ממערך
x = torch.tensor([1, 2, 3])
print("Tensor from list:", x)

# טנסור מלא באפסים
zeros = torch.zeros(2, 3)
print("Zeros tensor:", zeros)

# טנסור מלא באחדות
ones = torch.ones(4, 2)
print("Ones tensor:", ones)

# טנסור עם ערכים אקראיים
random_tensor = torch.rand(3, 3)
print("Random tensor:", random_tensor)

# טנסור עם ערכים רציפים
sequence = torch.arange(0, 10)
print("Sequence tensor:", sequence)

# טנסור מסוג float
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print("Float tensor:", float_tensor)

# העברת טנסור ל-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_on_gpu = torch.ones(3, 3).to(device)
print("Tensor on GPU:", tensor_on_gpu)

# פעולות על טנסורים
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([4, 5, 6], dtype=torch.float32)

# חיבור
z = x + y
print("Addition:", z)

# חיסור
z = x - y
print("Subtraction:", z)

# כפל איברי
z = x * y
print("Element-wise multiplication:", z)

# חילוק
z = x / y
print("Division:", z)

# כפל מטריצות
a = torch.rand(2, 3)
b = torch.rand(3, 4)
c = torch.matmul(a, b)
print("Matrix multiplication:", c)

# גודל הטנסור
x = torch.rand(3, 4)
print("Tensor shape:", x.shape)

# שינוי מבנה הטנסור (reshape)
x = torch.rand(6)
x = x.view(2, 3)
print("Reshaped tensor:", x)

# חישוב ממוצע, סכום, וערכים מקסימליים
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# ממוצע
mean = x.mean()
print("Mean:", mean)

# סכום
sum_x = x.sum()
print("Sum:", sum_x)

# ערך מקסימלי
max_x = x.max()
print("Max:", max_x)

# מעקב אחר גרדיאנטים
x = torch.ones(2, 2, dtype=torch.float32, requires_grad=True)
z = x.mean()  # חישוב z
z.backward()  # חישוב גרדיאנטים
print("Gradient of x:", x.grad)

shape = (2, 3)   # type <class 'tuple'> כדאי פשוט לתת ערך עבור shape
random_tensor = torch.rand(shape)
rand_int_tensor = torch.randint(0,9,shape)  #ערכים שלמים רנדומאליים מ0-9t
full_tensor = torch.full(shape,0.5)     #מכניס את הערך מספר מוגדר לכל הטנסור
range_tensor = torch.range(0,9) #ערכים  מ0 עד 9
# range_tensor = torch.range(9)ניתן לרשום גם כ
print(range_tensor)
print()