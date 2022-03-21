import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
import math
import os
import cv2


# preprocess a batch of images
def preprocess(x, mode='imagenet'):
    if mode:
        if 'imagenet' in mode:
            return preprocess_input(x)
        elif 'normalize' in mode:
            return  x.astype(np.float32) / 127.5 - 1
    else:
        return x

# applies to a lerarning rate tensor (lr) a decay schedule, the polynomial decay
def lr_decay(lr, init_learning_rate, end_learning_rate, epoch, total_epochs, power=0.9):
    # lr.assign(
    #     (init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate)
    pass

# converts a list of arrays into a list of tensors
def convert_to_tensors(list_to_convert):
    if list_to_convert != []:
        return [torch.tensor(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:])
    else:
        return []



# # Erase the elements if they are from ignore class. returns the labesl and predictions with no ignore labels
# TODO: check if this is correct
def erase_ignore_pixels(labels, predictions, mask):
    # indices = torch.squeeze(mask[mask > 0])  # not ignore labels
    # labels = torch.gather(labels, indices).type(torch.int64)
    # predictions = torch.gather(predictions, indices)
    # mask = mask[:,None]
    # mask = mask.expand(-1, 6)

    labels = labels * mask
    predictions = predictions * mask
    return labels, predictions


# generate and write an image into the disk
def generate_image(image_scores, output_dir, dataset, loader, train=False):
    # Get image name
    if train:
        list = loader.image_train_list
        index = loader.index_train
    else:
        list = loader.image_test_list
        index = loader.index_test

    dataset_name = dataset.split('/')
    if dataset_name[-1] != '':
        dataset_name = dataset_name[-1]
    else:
        dataset_name = dataset_name[-2]

    # Get output dir name
    out_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write it
    image = np.argmax(image_scores, 2)
    name_split = list[index - 1].split('/')
    name = name_split[-1].replace('.jpg', '.png').replace('.jpeg', '.png')
    cv2.imwrite(os.path.join(out_dir, name), image)

def inference(model, batch_images, n_classes, flip_inference=True, scales=[1], preprocess_mode=None):
    x = preprocess(batch_images, mode=preprocess_mode)
    # [x] = convert_to_tensors([x])
    x = torch.permute(x, (0, 3, 1, 2))
    # TODO: 3 dimension?
    x = x[:, 0:3, :, :]
    # creates the variable to store the scores
    y_ = convert_to_tensors([np.zeros((x.shape[0], n_classes, x.shape[2], x.shape[3]), dtype=np.float32)])[0]

    for scale in scales:
        # scale the image
        x_scaled = fn.resize(x, (x.shape[2]* scale, x.shape[3] * scale),
                            interpolation=T.InterpolationMode.BILINEAR)
        y_scaled = model(x_scaled, training=False)
        #  rescale the output
        y_scaled = fn.resize(y_scaled, (x.shape[2], x.shape[3]),
                                          interpolation=T.InterpolationMode.BILINEAR)
        # get scores
        y_scaled = F.softmax(y_scaled)

        if flip_inference:
            # calculates flipped scores
            y_flipped_ = torch.flip(model(torch.flip(x_scaled), training=False))
            # resize to rela scale
            y_flipped_ = fn.resize(y_flipped_, (x.shape[1].value, x.shape[2]),
                                                interpolation=T.InterpolationMode.BILINEAR)
            # get scores
            y_flipped_score = F.softmax(y_flipped_)

            y_scaled += y_flipped_score

        y_ += y_scaled

    return y_

# get accuracy and miou from a model
def get_metrics(loader, model, n_classes, train=True, flip_inference=False, scales=[1], write_images=False,
                preprocess_mode=None):
    if train:
        loader.index_train = 0
    else:
        loader.index_test = 0

    # accuracy = tf.metrics.Accuracy()
    conf_matrix = np.zeros((n_classes, n_classes))
    if train:
        samples = len(loader.image_train_list)
    else:
        samples = len(loader.image_test_list)

    for step in range(samples):  # for every batch
        x, y, mask = loader.get_batch(size=1, train=train, augmenter=False)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        mask = torch.from_numpy(mask)
        # y = torch.permute(y, (0, 3, 1, 2))
        # mask = torch.permute(mask, (0, 1, 2))
        
        # [y] = convert_to_tensors([y])
        y_ = inference(model, x, n_classes, flip_inference, scales, preprocess_mode=preprocess_mode)

        # generate images
        if write_images:
            generate_image(y_[0,:,:,:], 'images_out', loader.dataFolderPath, loader, train)

        # Rephape
        y_ = torch.permute(y_, (0, 2, 3, 1))
        y = torch.reshape(y, [y.shape[1] * y.shape[2] * y.shape[0], y.shape[3]])
        y_ = torch.reshape(y_, [y_.shape[1] * y_.shape[2] * y_.shape[0], y_.shape[3]])
        mask = torch.reshape(mask, [mask.shape[1] * mask.shape[2] * mask.shape[0]])

        labels, predictions = erase_ignore_pixels(labels=torch.argmax(y, 1), predictions=torch.argmax(y_, 1), mask=mask)
        acc = (labels == predictions).sum() / labels.size(0)
        
        conf_matrix += confusion_matrix(labels.numpy(), predictions.numpy(), labels=range(0, n_classes))

    # get the train and test accuracy from the model
    return acc.item(), compute_iou(conf_matrix)

# computes the miou given a confusion amtrix
def compute_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    IoU[np.isnan(IoU)] = 0
    print(IoU)
    miou = np.mean(IoU)
    '''
    print(ground_truth_set)
    miou_no_zeros=miou*len(ground_truth_set)/np.count_nonzero(ground_truth_set)
    print ('Miou without counting classes with 0 elements in the test samples: '+ str(miou_no_zeros))
    '''
    return miou
