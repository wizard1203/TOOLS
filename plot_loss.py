# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

#OUTPUTPATH='/media/sf_Shared_Data/tmp/p2psgd'
OUTPUTPATH='./'
max_epochs = 3000
EPOCH = True
FONTSIZE=18

fig, ax = plt.subplots(1,1,figsize=(5,3.8))
ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstman4': 'LSTM-AN4'
        }

def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def seconds_between_datetimestring(a, b):
    a = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    b = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    delta = b - a 
    return delta.days*86400+delta.seconds
sbd = seconds_between_datetimestring

def get_loss(line, isacc=False):
    if EPOCH:
        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('loss: ') > 0
        #if line.find('Epoch') > 0 and line.find('loss:') > 0 and not line.find('acc:')> 0:
        if line.find('Epoch') > 0 and valid: 
            items = line.split(' ')
            loss = float(items[-1])
            # print(loss)
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    else:
        if line.find('average forward') > 0:
            items = line.split('loss:')[1]
            loss = float(items[1].split(',')[0])
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    return None, None

def read_losses_from_log(logfile, isacc=False):
    f = open(logfile)
    print(logfile)
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None
    # max_epochs = max_epochs
    counter = 0
    for line in f.readlines():
        #if line.find('average forward') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
        if line.find('Epoch') > 0 and valid:
        #if not time0 and line.find('INFO [  100]') > 0:
            # print(line)
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            if not time0:
                time0 = t
        if line.find('lr: ') > 0:
            try:
                lr = float(line.split(',')[-2].split('lr: ')[-1])
                lrs.append(lr)
            except:
                pass
        if line.find('average delay: ') > 0:
            delay = int(line.split(':')[-1])
            average_delays.append(delay)
        loss, t = get_loss(line, isacc)
        # print(loss)
        if loss and t:
            # print(logfile, loss)
            counter += 1
            losses.append(loss)
            times.append(t)
        if counter > max_epochs:
            break

        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        #    items = line.split(' ')
        #    loss = float(items[-1])
        #    #items = line.split('loss:')[1]
        #    #loss = float(items[1].split(',')[0])

        #    losses.append(loss)
        #    t = line.split(' I')[0].split(',')[0]
        #    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        #    times.append(t)
    f.close()
    if not EPOCH:
        print('not find epoch')
        average_interval = 10
        times = [times[t*average_interval] for t in range(len(times)/average_interval)]
        losses = [np.mean(losses[t*average_interval:(t+1)*average_interval]) for t in range(len(losses)/average_interval)]
    print(losses)
    if len(times) > 0:
        t0 = time0 if time0 else times[0] #times[0]
        for i in range(0, len(times)):
            delta = times[i]- t0
            times[i] = delta.days*86400+delta.seconds
    return losses, times, average_delays, lrs

def read_norm_from_log(logfile):
    f = open(logfile)
    means = []
    stds = []
    for line in f.readlines():
        if line.find('gtopk-dense norm mean') > 0:
            items = line.split(',')
            mean = float(items[-2].split(':')[-1])
            std = float(items[--1].split(':')[-1])
            means.append(mean)
            stds.append(std)
    print('means: ', means)
    print('stds: ', stds)
    return means, stds

def plot_loss(logfile, label, isacc=False, title='ResNet-20', scale=None, comm_ratio=None):
    losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    norm_means, norm_stds = read_norm_from_log(logfile)
    new_losses = []
    if scale:
        for item in losses:
            for i in range(scale):
                new_losses.append(item)
        losses = new_losses

    # print(losses, times, average_delays, lrs)
    #print('times: ', times)
    #print('Learning rates: ', lrs)
    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    #plt.plot(losses, label=label, marker='o')
    #plt.xlabel('Epoch')
    #plt.title('ResNet-20 loss')
    if isacc:
        ax.set_ylabel('Top-1 Validation Accuracy')
    else:
        ax.set_ylabel('Training loss')
    #plt.title('ResNet-50')
    ax.set_title(get_real_title(title))
    marker = markeriter.next()
    color = coloriter.next()
    #print('marker: ', marker)
    #ax.plot(losses[0:180], label=label, marker=marker, markerfacecolor='none')
    if comm_ratio:
        model_size = 6653628.00
        x = np.arange(len(losses))
        x = comm_ratio * model_size * x
        ax.set_xlabel('Total communication size')
    else:
        x = np.arange(len(losses))
        ax.set_xlabel('Epoch')
    ax.plot(x, losses, label=label, marker=marker, markerfacecolor='none', color=color)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    if comm_ratio:
        ax.set_xlabel('Communication size (Bytes)')
    else:
        ax.set_xlabel('Epoch')
    #plt.plot(times, losses, label=label, marker=markeriter.next())
    #plt.xlabel('Time [s]')
    ax.grid(linestyle=':')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr
        #for i in lr_indexes:
        #    if i < len(losses):
        #        ls = losses[i]
        #        ax.text(i, ls, 'lr=%f'%lrs[i])
    u.update_fontsize(ax, FONTSIZE)

def plot_loss_with_host(hostn, nworkers, hostprefix, baseline=False):
    if not baseline or nworkers == 64:
        port = 5922
    else:
        port = 5945
    for i in range(hostn, hostn+1):
        for j in range(2, 3):
            host='%s%d-%d'%(hostprefix, i, port+j)
            if baseline:
                logfile = './ad-sgd-%dn-%dw-logs/'%(nworkers/4, nworkers)+host+'.log'
            else:
                logfile = './%dnodeslogs/'%nworkers+host+'.log'
                if nworkers == 256 and hostn < 48:
                    host='%s%d.comp.hkbu.edu.hk-%d'%(hostprefix, i, port+j)
                    logfile = './%dnodeslogs/'%nworkers+host+'.log'
                #csr42.comp.hkbu.edu.hk-5922.log
                #logfile = './%dnodeslogs-w/'%nworkers+host+'.log'
            label = host+' ('+str(nworkers)+' workers)'
            if baseline:
                label += ' Baseline'
            plot_loss(logfile, label) 

def resnet20():
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    #plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Allreduce', prefix='allreduce-baseline-wait-dc1-model-debug', nsupdate=1)
    plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', '(Ref 1/4 data)', prefix='compression-modele',sparsity=0.01)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Spar', prefix='compression-dc1-model-debug',sparsity=0.999)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Gradients', prefix='adpsgd-dc1-grad-debug',sparsity=None)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Gradients Sync', prefix='adpsgd-wait-dc1-grad-debug',sparsity=None)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Sequence d=0.1, lr=0.01', prefix='allreduce-comp-sequence-baseline-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu11', 'Sequence d=0.1, lr=0.1, wp', prefix='allreduce-comp-sequence-baseline-gwarmup-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.01, 'gpu17', 'Sequence d=0.1, lr=0.01, wp', prefix='allreduce-comp-sequence-baseline-gwarmup-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    #plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Sequence density=0.1', prefix='allreduce-comp-sequence-baseline-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.01, force_legend=True)

def vgg16():
    #plot_with_params('vgg16', 4, 32, 0.1, 'gpu17', 'Allreduce', prefix='allreduce')
    #plot_with_params('vgg16', 4, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.95)
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.98)
    plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.01, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-wait-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'MGD', 'ADPSGD ', prefix='baseline-modelmgd', title='VGG16')
    plot_with_params('vgg16', 16, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.1, 'gpu20', 'ADPSGD ', prefix='baseline-modelk80', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.0005, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 8, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 16, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')

def mnistnet():
    plot_with_params('mnistnet', 100, 50, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 512, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 64, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')


def mnistflnet():
    prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'
    plot_with_params('mnistflnet', 100, 50, 0.1, 'FedAvg', machine='csrlogs',  prefix='baseline-gwarmup-wait-dc1-model-fl', title='MNIST CNN')

    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='gpuhome_fl',  prefix='baseline-modelhpcl', title='MNIST CNN')
    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='gpuhomedc_p2p', prefix='baseline-modelhpcl', title='MNIST CNN')
    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='gpuhomedc2_p2p', prefix='baseline-modelhpcl', title='MNIST CNN')
    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='hswlogs', prefix='baseline-modelhpcl', title='MNIST CNN')


def cifar10flnet():
    plot_with_params('cifar10flnet', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='CIFAR-10 CNN')



def plot_one_worker():
    def _plot_with_params(bs, lr, isacc=True):
        logfile = './logs/resnet20/accresnet20-bs%d-lr%s.log' % (bs, str(lr))
        t = '(lr=%.4f, bs=%d)'%(lr, bs)
        plot_loss(logfile, t, isacc=isacc, title='resnet20') 
    _plot_with_params(32, 0.1)
    _plot_with_params(32, 0.01)
    _plot_with_params(32, 0.001)
    _plot_with_params(64, 0.1)
    _plot_with_params(64, 0.01)
    _plot_with_params(64, 0.001)
    _plot_with_params(128, 0.1)
    _plot_with_params(128, 0.01)
    _plot_with_params(128, 0.001)
    _plot_with_params(256, 0.1)
    _plot_with_params(256, 0.01)
    _plot_with_params(256, 0.001)
    _plot_with_params(512, 0.1)
    _plot_with_params(512, 0.01)
    _plot_with_params(512, 0.001)
    _plot_with_params(1024, 0.1)
    _plot_with_params(1024, 0.01)
    _plot_with_params(1024, 0.001)
    _plot_with_params(2048, 0.1)

def resnet50():
    plot_loss('baselinelogs/accresnet50-lr0.01-c40,70.log', 'allreduce 4 GPUs', isacc=False, title='ResNet-50') 

    plot_with_params('resnet50', 8, 64, 0.01, 'gpu10', 'allreduce 8 GPUs', prefix='allreduce-debug')
    plot_with_params('resnet50', 8, 64, 0.01, 'gpu16', 'ADPSGD', prefix='baseline-dc1-modelk80')

def plot_norm_diff():
    network = 'resnet20'
    bs =32
    #network = 'vgg16'
    #bs = 128
    path = './logs/allreduce-comp-gtopk-baseline-gwarmup-dc1-model-normtest/%s-n4-bs%d-lr0.1000-ns1-sg1.50-ds0.001' % (network,bs)
    epochs = 80
    arr = None
    arr2 = None
    arr3 = None
    for i in range(1, epochs):
        fn = '%s/gtopknorm-rank0-epoch%d.npy' % (path, i)
        fn2 = '%s/randknorm-rank0-epoch%d.npy' % (path, i)
        fn3 = '%s/upbound-rank0-epoch%d.npy' % (path, i)
        fn4 = '%s/densestd-rank0-epoch%d.npy' % (path, i)
        if arr is None:
            arr = np.load(fn)
            arr2 = np.load(fn2)
            arr3 = np.load(fn3)
            arr4 = np.load(fn4)
        else:
            arr = np.concatenate((arr, np.load(fn)))
            arr2 = np.concatenate((arr2, np.load(fn2)))
            arr3 = np.concatenate((arr3, np.load(fn3)))
            arr4 = np.concatenate((arr4, np.load(fn4)))
    #plt.plot(arr-arr2, label='||x-gtopK(x)||-||x-randomK(x)||')
    plt.plot(arr4, label='Gradients std')
    plt.xlabel('# of iteration')
    #plt.ylabel('||x-gtopK(x)||-||x-randomK(x)||')
    plt.title(network)
    #plt.plot(arr2, label='||x-randomK(x)||')
    #plt.plot(arr3, label='(1-K/n)||x||')

def loss(network):
    # Convergence
    gtopk_name = 'gTopKAllReduce'
    dense_name = 'DenseAllReduce'

    # resnet20
    if network == 'resnet20':
        plot_with_params(network, 4, 32, 0.1, 'gpu13', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'vgg16':
        # vgg16
        plot_with_params(network, 4, 128, 0.1, 'gpu13', dense_name, prefix='allreduce-baseline-dc1-model-icdcs', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu10', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'alexnet':
        plot_with_params(network, 8, 256, 0.01, 'gpu20', dense_name, prefix='allreduce-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, force_legend=True)
        plot_with_params(network, 8, 256, 0.01, 'gpu20', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'resnet50':
        plot_with_params(network, 8, 64, 0.01, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)
    #todo zhtang an4 ==============
    elif network == 'lstman4':
        plot_with_params(network, 8, 8, 1, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)

def communication_speed():
    pass

def plot_with_params(dnn, nworkers, bs, lr, legend, isacc=False, logfile='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False, scale=None, comm_ratio=None):
    # postfix='5922'
    # if prefix.find('allreduce')>=0:
    #     postfix='0'
    # if sparsity:
    #     logfile = './logs/%s/logs/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (machine, prefix, dnn, nworkers, bs, lr, sparsity)
    # elif nsupdate:
    #     logfile = './logs/%s/logs/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (machine, prefix, dnn, nworkers, bs, lr, nsupdate)
    # else:
    #     logfile = './logs/%s/logs/%s/%s-n%d-bs%d-lr%.4f' % (machine, prefix, dnn, nworkers, bs, lr)
    # if sg is not None:
    #     logfile += '-sg%.2f' % sg
    # if density is not None:
    #     logfile += '-ds%s' % str(density)
    # logfile += '%s.log' % (postfix)

    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    plot_loss(logfile, l, isacc=isacc, title=title, scale=scale, comm_ratio=comm_ratio)

def infocom2020_ced_fl_convergence(network, workers, if_iid):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_name = 'D-PSGD'
    CED_FL_name = 'CED-FL'
    # prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'
    if network == 'mnistflnet':
        if workers =='100':
            if if_iid :
                plot_with_params('mnistflnet', 100, 50, 0.1, FedAvg_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/hswlogs/logs/baseline-gwarmup-wait-dc1-model-fl/mnistflnet-n100-bs50-lr0.1000/hsw224-8923.log', force_legend=True, scale=2)
                plot_with_params('mnistflnet', 100, 50, 0.1, FedAvg_sparse_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/csrlogs/logs/compression-gwarmup-wait-dc1-model-fl/mnistflnet-n100-bs50-lr0.1000-s0.99000/csr51-6922.log', force_legend=True, scale=2)
                plot_with_params('mnistflnet', 100, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/hswlogs/logs/baseline-wait-dc1-model-dc/mnistflnet-n100-bs50-lr0.1000/hsw224-19622.log', force_legend=True, scale=10)
                plot_with_params('mnistflnet', 100, 50, 0.1, CED_FL_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/hswlogs/logs/compression-wait-dc1-model-dc/mnistflnet-n100-bs50-lr0.1000-s0.99000/hsw224-19622.log', force_legend=True)

            else:
                plot_with_params('mnistflnet', 100, 50, 0.01, FedAvg_name, isacc=True, title='MNIST CNN NON-IID',
                    logfile='logs/csrlogs/logs/baseline-gwarmup-wait-noniid-dc1-model-fl/mnistflnet-n100-bs50-lr0.0100_noniid2/csr51-6922.log', force_legend=True, scale=2)
                plot_with_params('mnistflnet', 100, 50, 0.01, FedAvg_sparse_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/hswlogs/logs/compression-wait-noniid-dc1-model-fl/mnistflnet-n100-bs50-lr0.0100-s0.90000-baseline1/hsw224-8923.log', force_legend=True, scale=2)
                plot_with_params('mnistflnet', 100, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/hswlogs/logs/baseline-wait-noniid-2samples-dc1-model-dc/mnistflnet-n100-bs50-lr0.1000/hsw224-19622.log', force_legend=True)
                plot_with_params('mnistflnet', 100, 50, 0.01, CED_FL_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/hswlogs/logs/compression-wait-noniid-2samples-dc1-model-dc/mnistflnet-n100-bs50-lr0.0100-s0.90000/hsw224-19622.log', force_legend=True)

        elif workers == '10':
            pass   
            if if_iid :
                pass
            else:
                pass

    elif network == 'cifar10flnet':
        if if_iid:
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhome_fl/logs/baseline-gwarmup-wait-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0100-baseline/gpu15-6923.log', force_legend=True, scale=2)
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_sparse_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhome_fl/logs/compression-gwarmup-wait-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0100-s0.99000-baseline/gpu15-6923.log', force_legend=True, scale=2)
            plot_with_params('cifar10flnet', 10, 50, 0.01, DPSGD_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhomedc2_p2p/logs/baseline-wait-dc1-model-dc/cifar10flnet-n10-bs100-lr0.0100/gpu17-9622.log', force_legend=True)
            plot_with_params('cifar10flnet', 10, 100, 0.01, CED_FL_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/hswlogs/logs/compression-wait-dc1-model-dc/cifar10flnet-n10-bs100-lr0.0100-s0.99000/hsw224-19622.log', force_legend=True)
        else:
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_name, isacc=True, title='CIFAR-10 CNN NON-IID',
                logfile='logs/gpuhome_fl/logs/baseline-gwarmup-wait-noniid-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0010-07-23-baseline/gpu15-6923.log', force_legend=True, scale=2)
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_sparse_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhome_fl/logs/compression-gwarmup-wait-noniid-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0100-s0.90000/gpu15-6923.log', force_legend=True, scale=2)
            plot_with_params('cifar10flnet', 10, 50, 0.01, DPSGD_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhomedc_p2p/logs/baseline-wait-noniid-dc1-model-dc/cifar10flnet-n10-bs100-lr0.0100/gpu10-8222.log', force_legend=True)
            plot_with_params('cifar10flnet', 10, 100, 0.01, CED_FL_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhomedc2_p2p/logs/compression-wait-noniid-dc1-model-dc/mnistflnet-n10-bs50-lr0.1000-s0.90000/gpu17-9622.log', force_legend=True)
 
def infocom2020_ced_fl_communication(network, workers, if_iid):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_name = 'D-PSGD'
    CED_FL_name = 'CED-FL'
    comm_ratio = {
        'FedAvg': 2,
        'S-FedAvg': 1+2*(1-0.99),
        'D-PSGD': 2*12,
        'CED-FL': 2*(1-0.99)*12,
    }
    # prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'
    if network == 'mnistflnet':
        if workers =='100':
            if if_iid :
                plot_with_params('mnistflnet', 100, 50, 0.1, FedAvg_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/hswlogs/logs/baseline-gwarmup-wait-dc1-model-fl/mnistflnet-n100-bs50-lr0.1000/hsw224-8923.log', force_legend=True, scale=2, comm_ratio=2)
                plot_with_params('mnistflnet', 100, 50, 0.1, FedAvg_sparse_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/csrlogs/logs/compression-gwarmup-wait-dc1-model-fl/mnistflnet-n100-bs50-lr0.1000-s0.99000/csr51-6922.log', force_legend=True, scale=2, comm_ratio=1+2*(1-0.99))
                plot_with_params('mnistflnet', 100, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/hswlogs/logs/baseline-wait-dc1-model-dc/mnistflnet-n100-bs50-lr0.1000/hsw224-19622.log', force_legend=True, scale=10, comm_ratio=2*12)
                plot_with_params('mnistflnet', 100, 50, 0.1, CED_FL_name, isacc=True, title='MNIST CNN IID',
                    logfile='logs/hswlogs/logs/compression-wait-dc1-model-dc/mnistflnet-n100-bs50-lr0.1000-s0.99000/hsw224-19622.log', force_legend=True, comm_ratio=2*(1-0.99)*12)

            else:
                plot_with_params('mnistflnet', 100, 50, 0.01, FedAvg_name, isacc=True, title='MNIST CNN NON-IID',
                    logfile='logs/csrlogs/logs/baseline-gwarmup-wait-noniid-dc1-model-fl/mnistflnet-n100-bs50-lr0.0100_noniid2/csr51-6922.log', force_legend=True, scale=2, comm_ratio=2)
                plot_with_params('mnistflnet', 100, 50, 0.01, FedAvg_sparse_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/hswlogs/logs/compression-wait-noniid-dc1-model-fl/mnistflnet-n100-bs50-lr0.0100-s0.90000-baseline1/hsw224-8923.log', force_legend=True, scale=2, comm_ratio=1+2*(1-0.9))
                plot_with_params('mnistflnet', 100, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/hswlogs/logs/baseline-wait-noniid-2samples-dc1-model-dc/mnistflnet-n100-bs50-lr0.1000/hsw224-19622.log', force_legend=True, comm_ratio=2*12)
                plot_with_params('mnistflnet', 100, 50, 0.01, CED_FL_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/hswlogs/logs/compression-wait-noniid-2samples-dc1-model-dc/mnistflnet-n100-bs50-lr0.0100-s0.90000/hsw224-19622.log', force_legend=True, comm_ratio=2*(1-0.9)*12)

        elif workers == '25':
            if if_iid :
                pass
            else:
                # plot_with_params('mnistflnet', 25, 50, 0.01, FedAvg_name, isacc=True, title='MNIST CNN NON-IID',
                #     logfile='logs/csrlogs/logs/baseline-gwarmup-wait-noniid-dc1-model-fl/mnistflnet-n100-bs50-lr0.0100_noniid2/csr51-6922.log', force_legend=True, scale=2, comm_ratio=2)
                plot_with_params('mnistflnet', 25, 50, 0.1, FedAvg_sparse_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/gpuhome_fl/logs/compression-wait-noniid-dc1-model-fl/mnistflnet-n25-bs50-lr0.1000-s0.70000/gpu10-6923.log', force_legend=True, scale=2, comm_ratio=1+2*(1-0.7))
                plot_with_params('mnistflnet', 25, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/gpuhomedc_p2p/logs/baseline-wait-noniid-dc1-model-dc/mnistflnet-n25-bs50-lr0.1000/gpu15-8222.log', force_legend=True, comm_ratio=2*12)
                plot_with_params('mnistflnet', 25, 50, 0.1, CED_FL_name+'-s0.7', isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/gpuhomedc_p2p/logs/compression-wait-noniid-dc1-model-dc/mnistflnet-n25-bs50-lr0.1000-s0.70000/gpu10-8222.log', force_legend=True, scale=10, comm_ratio=2*(1-0.7)*12)
                plot_with_params('mnistflnet', 25, 50, 0.1, CED_FL_name+'-s0.9', isacc=True, title='MNIST CNN NON_IID',
                    logfile='logs/gpuhomedc_p2p/logs/compression-wait-noniid-dc1-model-dc/mnistflnet-n25-bs50-lr0.0010-s0.90000/gpu10-8222.log', force_legend=True, scale=10, comm_ratio=2*(1-0.9)*12)



    elif network == 'cifar10flnet':
        if if_iid:
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhome_fl/logs/baseline-gwarmup-wait-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0100-baseline/gpu15-6923.log', force_legend=True, scale=2, comm_ratio=2)
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_sparse_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhome_fl/logs/compression-gwarmup-wait-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0100-s0.99000-baseline/gpu15-6923.log', force_legend=True, scale=2, comm_ratio=1+2*(1-0.99))
            plot_with_params('cifar10flnet', 10, 50, 0.01, DPSGD_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhomedc2_p2p/logs/baseline-wait-dc1-model-dc/cifar10flnet-n10-bs100-lr0.0100/gpu17-9622.log', force_legend=True, comm_ratio=2*50)
            plot_with_params('cifar10flnet', 10, 100, 0.01, CED_FL_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/hswlogs/logs/compression-wait-dc1-model-dc/cifar10flnet-n10-bs100-lr0.0100-s0.99000/hsw224-19622.log', force_legend=True, comm_ratio=2*(1-0.99)*50)
        else:
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_name, isacc=True, title='CIFAR-10 CNN NON-IID',
                logfile='logs/gpuhome_fl/logs/baseline-gwarmup-wait-noniid-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0010-07-23-baseline/gpu15-6923.log', force_legend=True, scale=2, comm_ratio=2)
            plot_with_params('cifar10flnet', 10, 100, 0.01, FedAvg_sparse_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhome_fl/logs/compression-gwarmup-wait-noniid-dc1-model-fl/cifar10flnet-n10-bs100-lr0.0100-s0.90000/gpu15-6923.log', force_legend=True, scale=2, comm_ratio=1+2*(1-0.9))
            plot_with_params('cifar10flnet', 10, 50, 0.01, DPSGD_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhomedc_p2p/logs/baseline-wait-noniid-dc1-model-dc/cifar10flnet-n10-bs100-lr0.0100/gpu10-8222.log', force_legend=True, comm_ratio=2*50)
            plot_with_params('cifar10flnet', 10, 100, 0.01, CED_FL_name, isacc=True, title='CIFAR-10 CNN IID',
                logfile='logs/gpuhomedc2_p2p/logs/compression-wait-noniid-dc1-model-dc/mnistflnet-n10-bs50-lr0.1000-s0.90000/gpu17-9622.log', force_legend=True, comm_ratio=2*(1-0.9)*50)


def infocom2020_ced_fl():
    def convergence():
        # network = 'mnistflnet'
        # workers = '100'
        # if_iid = True

        # network = 'mnistflnet'
        # workers = '100'
        # if_iid = False


        network = 'cifar10flnet'
        workers = '10'
        if_iid = True

        # network = 'cifar10flnet'
        # workers = '10'
        # if_iid = False

        # infocom2020_ced_fl_convergence(network=network, workers=workers, if_iid=if_iid)

        ax.set_xlim(xmin=-1)
        ax.legend(fontsize=FONTSIZE)
        plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
        #plt.savefig('%s/%s_convergence.pdf' % (OUTPUTPATH, network))
        plt.show()

    def communication_speed():
        # network = 'mnistflnet'
        # workers = '100'
        # if_iid = True

        # network = 'mnistflnet'
        # workers = '100'
        # if_iid = False

        network = 'mnistflnet'
        workers = '25'
        if_iid = False

        # network = 'cifar10flnet'
        # workers = '10'
        # if_iid = True

        # network = 'cifar10flnet'
        # workers = '10'
        # if_iid = False

        # infocom2020_ced_fl_communication(network=network, workers=workers, if_iid=if_iid)

        ax.set_xlim(xmin=-1)
        # plt.legend()
        ax.legend(fontsize=FONTSIZE,loc='lower right')
        plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
        #plt.savefig('%s/%s_convergence.pdf' % (OUTPUTPATH, network))
        plt.show()



    # convergence()
    communication_speed()




if __name__ == '__main__':
    #resnet20()
    infocom2020_ced_fl()