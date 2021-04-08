import numpy as np

a = {
    "name" : "samiksha",
    "age" : 27,
}

b = {
    "name" : "shrek",
    "age": 30,
}

c = []
c.append(a)
c.append(b)




# x = [["sami", "shrek", "team"], [27, 30, 57]]
# x = np.array(x)
# for a in range(x.shape[1]):
#     print(x[0][a], x[1][a])

my_list = []
x = np.array([["sami", 27], ["nari", 30], ["team", 57]])
for a in x:
    my_dict = {
        "name" : a[0],
        "age" : a[1],
    }
    my_list.append(my_dict)
print(my_list)





