import numpy as np
import torch
import torch.nn as nn
import os
import nets.Network as Segception
import utils.Loader as Loader
from utils.utils import preprocess, lr_decay, convert_to_tensors, get_metrics
import argparse
import cv2

import setproctitle
import pdb
setproctitle.setproctitle('SemSeg@xinshiduo')

# enable eager mode
# tf.enable_eager_execution()
torch.manual_seed(7)
# TODO: figure out why random_seed
# np.random_seed(7)


# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=None, lr=None, init_lr=2e-4,
          saver=None, variables_to_optimize=None, evaluation=True, name_best_model = 'weights/best', preprocess_mode=None,
          optimizer=None, scheduler=None):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0

    for epoch in range(epochs):  # for each epoch
        # lr_decay(lr, init_lr, 1e-9, epoch, epochs - 1)  # compute the new lr
        scheduler.step()
        print('epoch: ' + str(epoch) + '. Learning rate: ' + str(lr))
        loss_all=0
        for step in range(steps_per_epoch):  # for every batch
            # get batch
            x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)

            x = preprocess(x, mode=preprocess_mode)
            [x, y, mask] = convert_to_tensors([x, y, mask])
            x = torch.permute(x, (0, 3, 1, 2)).cuda()
            y = torch.permute(y, (0, 3, 1, 2)).cuda()
            mask = torch.permute(mask, (0, 1, 2)).cuda()
            x = x[:, 0:3, :, :]
            y_, aux_y_ = model(x, aux_loss=True)  # get output of the model

            # loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)  # compute loss
            # loss_aux = tf.losses.softmax_cross_entropy(y, aux_y_, weights=mask)  # compute loss
            # y=torch.reshape(y,(y.shape[0]*y.shape[1],y.shape[2],y.shape[3]))
            # y_=torch.reshape(y_,(y_.shape[0]*y_.shape[1],y_.shape[2],y_.shape[3]))
            # aux_y_=torch.reshape(aux_y_,(aux_y_.shape[0]*aux_y_.shape[1],aux_y_.shape[2],aux_y_.shape[3]))
            target = torch.zeros((batch_size, y.shape[2], y.shape[3])).cuda()
            for i in range(y.shape[1]):
                target[torch.squeeze(y[:,i,:,:]==1)] = i
                
            # https://discuss.pytorch.org/t/cross-entropy-loss-error-on-image-segmentation/60194/12
            # TODO: Check if this is correct? add a softmax
            # y_ = y_.softmax(dim=1)
            loss = nn.functional.cross_entropy(y_, target.long())
            loss_aux = nn.functional.cross_entropy(aux_y_, target.long())
            loss = 1 * loss + 0.8 * loss_aux
            loss_all+=loss
            if show_loss: print('Training loss: ' + str(loss.numpy()))

            # Gets gradients and applies them
            # grads = g.gradient(loss, variables_to_optimize)
            # optimizer.apply_gradients(zip(grads, variables_to_optimize))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if evaluation:
            # get metrics
            #train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode)
            model.eval()
            test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                              scales=[1], preprocess_mode=preprocess_mode)
            model.train()
            #print('Train accuracy: ' + str(train_acc.numpy()))
            #print('Train miou: ' + str(train_miou))
            print('Test accuracy: ' + str(test_acc))
            print('Test miou: ' + str(test_miou))
            print('')
            with open('./res_10tr.txt','a') as f:
               
                f.write("%f,%f,%f" % (
                    loss_all,test_acc,test_miou
                ))
                f.write("\n")
            # save model if bet
            if test_miou > best_miou:
                best_miou = test_miou
                torch.save(model.state_dict(), './bestmodel.tar')
        else:
            torch.save(model.state_dict(), './bestmodel.tar')
            # model.load_state_dict(torch.load('./bestmodel.tar',map_location=device))

        loader.suffle_segmentation()  # sheffle trainign set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", help="Dataset path", default='test_data')
    parser.add_argument("--dataset", help="Dataset path", default="/data/xinshiduo/code/Ev-SegNet-master/dataset_our_codification/")
    parser.add_argument("--model_path", help="Model path", default='weights/model')
    parser.add_argument("--n_classes", help="number of classes to classify", default=6)
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("--epochs", help="number of epochs to train", default=100)
    parser.add_argument("--width", help="number of epochs to train", default=320)
    parser.add_argument("--height", help="number of epochs to train", default=320)
    parser.add_argument("--lr", help="init learning rate", default=1e-3)
    parser.add_argument("--n_gpu", help="number of the gpu", default=3)
    args = parser.parse_args()

    n_gpu = int(args.n_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)


    n_classes = int(args.n_classes)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    width =  int(args.width)
    height =  int(args.height)
    lr = float(args.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channels = 6 # input of 6 channels
    channels_image = 0
    channels_events = channels - channels_image
    folder_best_model = args.model_path
    name_best_model = os.path.join(folder_best_model,'best')
    dataset_path = args.dataset
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width, height=height, channels=channels_image, channels_events=channels_events)

    if not os.path.exists(folder_best_model):
        os.makedirs(folder_best_model)

    # build model and optimizer
    model = Segception.Segception_small(num_classes=n_classes, weights=None, input_shape=(None, None, channels))
    model=model.to(device)
    # import torchsummary as summary
    # print(summary(model, (3, 299, 299)))
    # optimizer
    # learning_rate = tfe.Variable(lr)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # TODO: ? align with lr_decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)


    # Init models (optional, just for get_params function)
    # init_model(model, input_shape=(batch_size, width, height, channels)) DEPRECATED

    # variables_to_restore = model.variables #[x for x in model.variables if 'block1_conv1' not in x.name]
    # variables_to_save = model.variables
    # TODO: check!!!
    # variables_to_optimize = model.variables
    variables_to_optimize = None

    # Init saver. can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)
    # saver_model = tfe.Saver(var_list=variables_to_save)
    # restore_model = tfe.Saver(var_list=variables_to_restore)
    PATH = args.model_path + 'best.pth'
    saver_model = torch.save(model.state_dict(), PATH)
    restore_model = torch.save(model.state_dict(), PATH)

    # restore if model saved and show number of params
    # restore_state(restore_model, name_best_model) DEPRECATED
    # get_params(model)  DEPRECATED

    train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=lr,
          init_lr=lr, saver=saver_model, variables_to_optimize=variables_to_optimize, name_best_model=name_best_model,
          evaluation=True, preprocess_mode=None, optimizer = optimizer, scheduler = scheduler)

    # Test best model
    print('Testing model')
    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1, 0.75, 1.5],
                                      write_images=False, preprocess_mode=None)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou))

    #train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode)
    #print('Train accuracy: ' + str(train_acc.numpy()))
    #print('Train miou: ' + str(train_miou))