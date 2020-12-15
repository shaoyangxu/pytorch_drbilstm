# from collections import Counter
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# train_data_path = "mydata/train_data.pkl"
# dev_data_path = "mydata/dev_data2.pkl"
# test_data_path = "mydata/test_data2.pkl"
#
import pickle
# with open(train_data_path,"rb") as pkl:
#     train_data = pickle.load(pkl)
#
# with open(dev_data_path,"rb") as pkl:
#     dev_data = pickle.load(pkl)
#
# with open(test_data_path, "rb") as pkl:
#     test_data = pickle.load(pkl)
#
# print(len(train_data["premises"]))
# print(len(dev_data["premises"]))
# print(len(test_data["premises"]))
#
# def label_distribution(data):
#     labels = data["labels"]
#     # '0 = entailment, 1 = neutral, 2 = contradiction'
#     e,n,c = 0,0,0
#     for i in labels:
#         if i == 0:
#             e += 1
#         elif i == 1:
#             n += 1
#         else:
#             c += 1
#     print(e,n,c)
# print("train")
# label_distribution(train_data)
# print("dev")
# label_distribution(dev_data)
# print("test")
# label_distribution(test_data)
#
# def sentence_length(data):
#     premises = data["premises"]
#     hypotheses = data["hypotheses"]
#     p_lens = []
#     for p in premises:
#         p_lens.append(len(p))
#     h_lens = []
#     for h in hypotheses:
#         h_lens.append(len(h))
#     plt.plot(list(Counter(p_lens).values()))
#     plt.savefig("p.jpg")
#     plt.plot(list(Counter(h_lens).values()))
#     plt.savefig("h.jpg")
#     # counter_p = Counter(p_lens)
#     # print(counter_p)
#
# def extract_data(data,num,name):
#     idss = data["ids"]
#     premises = data["premises"]
#     hypotheses = data["hypotheses"]
#     labels = data["labels"]
#     idx_list = [i for i in range(len(idss))]
#     import random
#     random.shuffle(idx_list)
#     i,p,h,l = [],[],[],[]
#     for j in range(num):
#         i.append(idss[j])
#         p.append(premises[j])
#         h.append(hypotheses[j])
#         l.append(labels[j])
#     new_dict = {}
#     new_dict["ids"] = i
#     new_dict["premises"] = p
#     new_dict["hypotheses"] = h
#     new_dict["labels"] = l
#     path = "mydata/{}.pkl".format(name)
#     with open(path, "wb") as pkl:
#         pickle.dump(new_dict, pkl)
# #
# # extract_data(train_data,8000,"train_data")
# # extract_data(dev_data,150,"dev_data2")
# # extract_data(test_data,150,"test_data2")
path1 = "test_data.pkl"
# ['ids':True, 'premises', 'hypotheses', 'le_premises', 'le_hypotheses', 'labels']
with open(path1,"rb") as pkl:
    a = pickle.load(pkl)['premises']

path2 = "preprocessed_data/test_data.pkl"
with open(path2, "rb") as pkl:
    b = pickle.load(pkl)['premises']
print(a==b)