#!/usr/bin/env python

import os
import sys
import time
import tensorflow as tf
import numpy as np
import collections
from bdgan_models_synth import BGAN
from bgan_util import SynthDataset, SynthDataset3, SynthDataset4, FigPrinter
import matplotlib.pyplot as plt
from sklearn import mixture
import seaborn as sns
from sklearn.decomposition import PCA

def get_session():
    global _SESSION
    if tf.get_default_session() is None:
        _SESSION = tf.InteractiveSession()
    else:
        _SESSION = tf.get_default_session()

    return _SESSION

def bgan_synth(dataset, synth_dataset, z_dim, clip, batch_size=128, num_iter=15000, 
               rpath="synth_results",
               base_learning_rate=0.00015, lr_decay=3., save_weights=False):
    bgan = BGAN([synth_dataset.x_dim], z_dim,
                synth_dataset.N,
                batch_size=batch_size, lr=0.0002, alpha=0.00012)
    print ("Starting session")
    session = get_session()
    tf.global_variables_initializer().run()
    print ("Starting training loop")
    num_train_iter = num_iter
    learning_rate = base_learning_rate
    np_samples = []
    start_time = time.time()
    for train_iter in range(num_train_iter):
        batch_z = np.random.normal(0., 1, [batch_size, z_dim])
        input_batch = synth_dataset.next_batch(batch_size)
        _, d_loss, d_real = session.run([bgan.d_optim_adam, bgan.d_loss, bgan.D], feed_dict={bgan.inputs: input_batch, bgan.z: batch_z, bgan.drop_prob:0.99,bgan.d_learning_rate: learning_rate})
        g_losses = []
        for gi in range(2):
            z = np.random.normal(0., 1, [batch_size, z_dim])
            _, gl, d_fake = session.run([bgan.g_optims_adam, bgan.g_loss, bgan.D_], feed_dict={bgan.z: batch_z, bgan.drop_prob:0.99, bgan.g_learning_rate: learning_rate})
        g_losses.append(gl)
        if (train_iter + 1) % 10000 == 0:
            fake_data=[]
            for i in range(10):
                sample_z = np.random.normal(0., 1, size=(1000, z_dim))
                sampled_data = session.run(bgan.generation["generators"][0], feed_dict={bgan.z: sample_z, bgan.drop_prob:0.99})
                fake_data.append(sampled_data)
            X_real = synth_dataset.next_batch(10000)
            X_sample = np.concatenate(fake_data)
            if(dataset=='synth'):
                pca = PCA(2)
                X_real = pca.fit_transform(X_real)   
                X_sample = pca.transform(X_sample)
            '''np_samples_ = X_sample[::1]
            fp = FigPrinter((1,2))
            xmin1 = np.min(X_real[:, 0]) - 0.5
            xmax1 = np.max(X_real[:, 0]) + 0.5
            xmin2 = np.min(X_real[:, 1]) - 0.5
            xmax2 = np.max(X_real[:, 1]) + 0.5
            fp.ax_arr[0].plot(X_real[:, 0], X_real[:, 1], '.r')
            fp.ax_arr[0].set_xlim([xmin1, xmax1]); fp.ax_arr[0].set_ylim([xmin2, xmax2])
            fp.ax_arr[1].plot(X_sample[:, 0], X_sample[:, 1], '.g')
            fp.ax_arr[1].set_xlim([xmin1, xmax1]); fp.ax_arr[1].set_ylim([xmin2, xmax2])
            fp.ax_arr[0].set_aspect('equal', adjustable='box')
            fp.ax_arr[1].set_aspect('equal', adjustable='box')
            fp.ax_arr[1].set_title("Iter %i" % (train_iter+1))            
            fp.print_to_file(os.path.join(rpath, "train_results_%i.png" % (train_iter+1)))
            bg_color  = sns.color_palette('Greens', n_colors=256)[0]
            ax2 = sns.kdeplot(np_samples_[:, 0], np_samples_[:, 1], shade=True, cmap='Greens', n_levels=20, clip=clip)
            ax2.set_facecolor(bg_color)'''
            #plt.savefig(os.path.join(rpath, "train_density_plot_bdgan_{}_{}.png".format(dataset,train_iter+1)))
    samples=[]
    end_time = time.time()
    train_time = end_time - start_time
    print(train_time)
    counter=0
    other_counter=0
    flag=1
    while(True):
        sample_z = np.random.normal(0., 1, size=(1, z_dim))
        ai, bi, sample = session.run([bgan.h0, bgan.drop, bgan.x], feed_dict={bgan.z: sample_z, bgan.drop_prob:0.99})
        other_counter = other_counter + 1
        if(np.all((bi == 0))):
            continue
        if(np.count_nonzero(bi)>1 or flag==0):
            samples.append(sample)
        if(len(samples)==9600):
            flag=0
        if(len(samples)>=10000):
            break
        counter = counter + 1
    samples2 = np.concatenate(samples)
    #print(counter)
    #print(other_counter)
    np.savetxt('original_{}.txt'.format(dataset),synth_dataset.next_batch(10000),delimiter=',')
    np.savetxt('bdgan_samples3_{}.txt'.format(dataset),samples2,delimiter=',')    

    bg_color  = sns.color_palette('Greens', n_colors=256)[0]
    ax2 = sns.kdeplot(samples2[:, 0], samples2[:, 1], shade=True, cmap='Greens', n_levels=20, clip=clip)
    ax2.set_facecolor(bg_color)
    plt.savefig(os.path.join(rpath, "density_plot_bdgan_{}.png".format(dataset)))

    g = (sns.jointplot(samples2[:,0],samples2[:,1],
                    color="k")
        .plot_joint(sns.kdeplot, zorder=0, n_levels=6))
    plt.savefig(os.path.join(rpath, "density_sc_plot_bdgan_{}.png".format(dataset)))
    
    fp = FigPrinter((1,2))
    xmin1 = np.min(X_real[:, 0]) - 0.5
    xmax1 = np.max(X_real[:, 0]) + 0.5
    xmin2 = np.min(X_real[:, 1]) - 0.5
    xmax2 = np.max(X_real[:, 1]) + 0.5
    fp.ax_arr[0].plot(X_real[:, 0], X_real[:, 1], '.r')
    fp.ax_arr[0].set_xlim([xmin1, xmax1]); fp.ax_arr[0].set_ylim([xmin2, xmax2])
    fp.ax_arr[1].plot(samples2[:, 0], samples2[:, 1], '.g')
    fp.ax_arr[1].set_xlim([xmin1, xmax1]); fp.ax_arr[1].set_ylim([xmin2, xmax2])
    fp.ax_arr[0].set_aspect('equal', adjustable='box')
    fp.ax_arr[1].set_aspect('equal', adjustable='box')
    fp.ax_arr[1].set_title("Iter %i" % (train_iter+1))            
    fp.print_to_file(os.path.join(rpath, "results_%i.png" % (10+1)))

    return {"data_real": synth_dataset.X,
            "z_dim": z_dim,
            "num_iter": num_iter}


if __name__ == "__main__":

    import argparse
    import time
    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN synthetic experiments')
    parser.add_argument('--x_dim',
                        type=int,
                        default=100,
                        help='dim of x for synthetic data')
    parser.add_argument('--z_dim',
                        type=int,
                        default=100,
                        help='dim of z for generator')
    parser.add_argument('--train_iter',
                        type=int,
                        default=70000,
                        help='no of GAN iterations')
    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='datasate name mnist etc.')
    parser.add_argument('--out_dir',
                        default="/tmp/synth_results",
                        help='path of where to store results')
    parser.add_argument('--random_seed',
                        type=int,
                        default=2,
                        help='set seed for repeatability')
    parser.add_argument('--save_weights',
                        action="store_true",
                        help='whether to save weight vectors')

    args = parser.parse_args()

    #if args.random_seed is not None:
        #print('seed is set')
        #np.random.seed(args.random_seed)
        #tf.set_random_seed(args.random_seed)

    if not os.path.exists(args.out_dir):
        print("Creating %s" % args.out_dir)
        os.makedirs(args.out_dir)

    results_path = os.path.join(args.out_dir, "experiment_%i" % (int(time.time())))
    os.makedirs(results_path)
    import pprint
    with open(os.path.join(results_path, "args.txt"), "w") as hf:
        hf.write("Experiment settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))

    if args.dataset == "synth":
        synth_d = SynthDataset()
        clip = None
    elif args.dataset == "2dring":
        synth_d = SynthDataset3()
        clip = [[-1.5, 1.5]]*2
    elif args.dataset == "2dgrid":
        synth_d = SynthDataset4()
        clip = [[-6, 6]]*2
    else:
        raise RuntimeError("invalid dataset %s" % args.dataset)

    results = bgan_synth(args.dataset, synth_d, args.z_dim, clip=clip, num_iter=args.train_iter, rpath=results_path, save_weights=args.save_weights)
    np.savez(os.path.join(results_path, "run_.npz"))




