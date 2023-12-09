#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, traffic_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update_dp import LocalUpdate_dp
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet
from models.Fed import FedAvg
from models.test import test_img
import loading_data as dataset


attack_attempted = 0
attack_successful = 0
phished = None
dp = None
accmulated_rewards = 0

def get_train_valid_loader(data_dir,
                           batch_size,
                           num_workers=0,
                           ):
    # Create Transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    dataset_train = dataset.BelgiumTS(
        root_dir=data_dir, train=True,  transform=transform)
    dataset_test = dataset.BelgiumTS(
        root_dir=data_dir, train=False,  transform=transform)

    # Load Datasets
    return dataset_train, dataset_test

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader('', batch_size=32, num_workers=0)
        if args.iid:
            dict_users = traffic_iid(dataset_train, args.num_users)
            print("this is unique user:", dict_users)
            print(len(dict_users[0]))
            print(len(dict_users[1]))
            print(len(dict_users[2]))
            print(len(dict_users[3]))
        else:
            exit('Error: only consider IID setting in Traffic')
    else:
        exit('Error: unrecognized dataset')
    #img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'LeNet' and args.dataset == 'traffic':
        net_glob = LeNet(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = np.sort(idxs_users)
        print("users:",idxs_users)

        #if random.random() < 1.0:
        ######attack and defense#######
        pd_nf = 0.9 #action0
        pd_acl = 0.5  #action 1

        pd_da = 0.5 #action2
        pd_su = 0.7 #action3

        #no defense --> action 4

        if phished == None:
            targeted_node = random.randint(0, 3)
            print("targeted node:",targeted_node)
        elif phished != None:
            targeted_node = phished
            print("targeted node:", targeted_node)

        if phished == None:
            attack_chosen = random.randint(0, 1) #0: dos, 1: phishing, 2: model poisoning
        elif phished != None:
            attack_chosen = 2
        print("attack_chosen:", attack_chosen)

        if random.random() < 1.0:
            attack_chosen = attack_chosen
            print("Correct: attack detected by NIDS:", attack_chosen)
        else:
            attack_chosen1 = [0, 1, 2]
            attack_chosen1.remove(attack_chosen)
            attack_chosen = random.choice(attack_chosen1)
            print("Incorrect: attack detected by NIDS:", attack_chosen)

        ####
        if random.random() < 0.8:
            print("attack launched")
        else:
            attack_chosen = None
            defense_chosen = 4
            print("no defense")
        #####

        if attack_attempted == 0 and attack_successful ==0:
            current_asr = 0
        else:
            current_asr = attack_successful/attack_attempted
        #state for DRL agent
        state = [attack_chosen, current_asr]
        #attack 0 --> dos
        #attack 1 --> phishing
        #attack 2 --> data poisoning

        print("state:", state)


        #defense action has to be determined by ƒthe DRL agent
        defense_chosen = random.randint(0, 4) #None
        print("defense action:",defense_chosen)
    ####
        #if attack_chosen == 0:
        #    defense_chosen = 0
        #if attack_chosen == 1:
        #    defense_chosen = 3
        #if attack_chosen == 2:
        #    defense_chosen = 0
        #print("defense action:", defense_chosen)
    ####

        if attack_chosen == 0: ##DoS
            if defense_chosen == 0:
                if random.random() <= pd_nf:
                    attack_attempted += 1
                    print("NF defense worked")
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    idxs_users = np.delete(idxs_users, targeted_node)
                    print(idxs_users)

            elif defense_chosen == 1:
                if random.random() <= pd_acl:
                    attack_attempted += 1
                    print("ACL defense worked")
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    idxs_users = np.delete(idxs_users, targeted_node)
                    print(idxs_users)

            else:#if defense_chosen == 4:
                attack_successful += 1
                attack_attempted += 1
                idxs_users = np.delete(idxs_users, targeted_node)
                print(idxs_users)

####
        if attack_chosen == 1: ##phshing
            if defense_chosen == 2:
                if random.random() <= pd_da:
                    attack_attempted += 1
                    phished = None
                    print("DA defense worked")
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    phished = targeted_node
                    print(idxs_users)

            elif defense_chosen == 3:
                if random.random() <= pd_su:
                    attack_attempted += 1
                    phished = None
                    print("SU defense worked")
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    phished = targeted_node
                    print(idxs_users)

            else:#if defense_chosen == 4:
                attack_successful += 1
                attack_attempted += 1
                phished = targeted_node
                print(idxs_users)


####
        if attack_chosen == 2: ##model poisoning
            if defense_chosen == 3:
                if random.random() <= pd_su:
                    attack_attempted += 1
                    print("SU defense worked")
                    phished = None
                    dp = False
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    dp = True
                    phished = targeted_node
                    print(idxs_users)

            elif defense_chosen == 0:
                if random.random() <= pd_nf:
                    attack_attempted += 1
                    print("NF defense worked")
                    phished = None
                    dp = False
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    dp = True
                    phished = targeted_node
                    print(idxs_users)

            elif defense_chosen == 1:
                if random.random() <= pd_acl:
                    attack_attempted += 1
                    print("ACL defense worked")
                    phished = None
                    dp = False
                    print(idxs_users)
                else:
                    attack_successful += 1
                    attack_attempted += 1
                    dp = True
                    phished = targeted_node
                    print(idxs_users)

            else:#if defense_chosen == 4:
                attack_successful += 1
                attack_attempted += 1
                dp = True
                phished = targeted_node
                print(idxs_users)
        #####end of attack and defense#####

        if attack_attempted == 0 and attack_successful == 0:
            next_asr = 0
        else:
            next_asr = attack_successful / attack_attempted
        print("curr_asr:", current_asr)
        print("next_asr:", next_asr)

        print("attack success:", attack_successful)
        print("attack attempt:", attack_attempted)

        rewards = current_asr-2*next_asr
        print("rewards:", rewards)
        #print(iter)
        accmulated_rewards += ((0.98)**iter) * rewards
        print("acculated_rewareds:",accmulated_rewards)

        for idx in np.sort(idxs_users):
            print("start training:", idx)


            if ((int(idx) == phished and dp == True)): #or int(idx) in fn):  #or int(idx) == victim_mp):#attack == 4)
                local = LocalUpdate_dp(args=args, dataset=dataset_train, idxs=dict_users[idx])  # set variable "local" as the class "LocalUpdate". dict_users is a dictionary of images that each client holds
                # print("this one:",dict_users)
                w, loss = local.train_dp(net=copy.deepcopy(net_glob).to(args.device))  # train each local client with global model

            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        ########dp
        if dp == True: #chaging global model in ccs
            print("dp performing")
            for k in w_glob.keys():
                w_glob[k] = torch.mul(w_glob[k], 0.8)  # -0.5)  #-0.8 is a good option it was -1.0 it was -0.1
        #######dp

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # testing
        net_glob.eval()
        acc_train, loss_train_updated, liness, correct_inst1, total_inst1 = test_img(net_glob, dataset_train, args)
        #train_accuracy.append(acc_train)
        acc_test, loss_test, lines, correct_inst, total_inst = test_img(net_glob, dataset_test, args)
        #test_accuracy.append(acc_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))


test_accuracy = int(acc_test)

file1 = open('random_prediction.txt', 'a')
s = str(test_accuracy)+"\n"
# Writing a string to file
file1.write(s)
# Writing multiple strings
# at a time
# Closing file
file1.close()


file2 = open('random_cumulreward.txt', 'a')
ss = str(accmulated_rewards)+"\n"
# Writing a string to file
file2.write(ss)
# Writing multiple strings
# at a time
# Closing file
file2.close()


file3 = open('random_asr.txt', 'a')
sss = str(next_asr)+"\n"
# Writing a string to file
file3.write(sss)
# Writing multiple strings
# at a time
# Closing file
file3.close()



#    # plot loss curve
#    plt.figure()
#    plt.plot(range(len(loss_train)), loss_train)
#    plt.ylabel('train_loss')
#    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
#    net_glob.eval()
#    acc_train, loss_train = test_img(net_glob, dataset_train, args)
#    acc_test, loss_test = test_img(net_glob, dataset_test, args)
#    print("Training accuracy: {:.2f}".format(acc_train))
#    print("Testing accuracy: {:.2f}".format(acc_test))

