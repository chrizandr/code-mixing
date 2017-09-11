"""Split the data into train and test according to the percentage."""

import json
from data_handler import read_data
import pdb


def write_to_file(data, filepath):
    """Write data to the file."""
    f = open(filepath, 'w')
    dump = json.dumps(data, indent=4, sort_keys=True)
    f.write(dump)


def find_distribution(data):
    """Find the distributions of classes in the data."""
    id_list = [[], [], []]
    positive = list()
    for Id,each in enumerate(data):
        if each['sentiment']==-1:
           id_list[2].append(Id)
        elif each['sentiment']==0 or each['sentiment']==1:
           id_list[each['sentiment']].append(Id)
    size = min(min(len(id_list[0]),len(id_list[1])),len(id_list[2]))
    print size
    nrmlzd_list = [id_list[0][0:size], id_list[1][0:size], id_list[2][0:size]]
    return nrmlzd_list, size


def divde_train_test(data):
    """Divide the data into train and test sets."""
    nrmlzd_list, size = find_distribution(data)
    train_size = int(0.8*size)
    test_size = int(0.2*size)
    print train_size, test_size
    train = []
    test = []
    train_id = []
    test_id = []
    counter = 0
    for each in nrmlzd_list:
        for i in range(0,train_size):
            data[each[i]]['id']=each[i]
            train.append(data[each[i]])
            train_id.append(each[i])
            #pdb.set_trace()
        for j in range(0, test_size):
            data[each[j]]['id']=each[j]
            test.append(data[each[j]])
            test_id.append(each[j])
    #pdb.set_trace()
    print len(train), len(test)
    return train, test


if __name__ == "__main__":
    data = read_data("final_codemixed.json")
    train, test = divde_train_test(data)
    write_to_file(train, "train_data.json")
    write_to_file(test, "test_data.json")
