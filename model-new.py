import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import numpy as np

# from video_data_loader import VideoDataLoader
from videoCap import Video

model_ctx = mx.cpu()
data_ctx = mx.cpu()

# define a convolutional neural network
def MKnet():
    net = gluon.nn.Sequential()
    
    with net.name_scope():
        # convolutional layers with pooling
        net.add(gluon.nn.Conv2D(channels=96, kernel_size=3, activation='relu'))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        net.add(gluon.nn.Conv2D(channels=48, kernel_size=3, activation='relu'))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        net.add(gluon.nn.Conv2D(channels=24, kernel_size=3, activation='relu'))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        net.add(gluon.nn.Conv2D(channels=20, kernel_size=3, activation='relu'))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        
        net.add(gluon.nn.Conv2D(channels=15, kernel_size=3, activation='relu'))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        
        # fully connected layers
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(.5))
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(.5))
        net.add(gluon.nn.Dense(2))
        
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
    return net

def evaluate_accuracy(model, dataIterator):
    accuracy = mx.metric.Accuracy()
    
    for batch in dataIterator:
        data = batch.data[0].as_in_context(mx.cpu())
        labels = batch.label[0].as_in_context(mx.cpu())
        outputs = model(data)
        predictions = nd.argmax(outputs, axis=1)
        
        accuracy.update(preds=predictions, labels=labels)

    dataIterator.reset()  
    return accuracy.get()[1]

def train(model, train_data, test_data, epochs, trainer, smce_loss):
    print("Starting training ...")
    for e in range(epochs):
        cumulative_loss = 0
        
        for batch in train_data:
            data = batch.data[0].as_in_context(mx.cpu())
            labels = batch.label[0].as_in_context(mx.cpu())
            
            with autograd.record():
                outputs = model(data)
                loss = smce_loss(outputs, labels)
                
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        train_accuracy = evaluate_accuracy(model, train_data)
        test_accuracy = evaluate_accuracy(model, test_data)

            
        loss_val = cumulative_loss / (batch.data[0].shape[0] * 32)
        print("Epoch %s | Loss: %s | Train_Acc: %s | Test_Acc: %s" %
             (e, loss_val, train_accuracy, test_accuracy))

        train_data.reset()
        test_data.reset()

def main():
    # video_files = [('./data/fast_lap_01.mp4', 0), ('./data/slow_lap_01.mp4', 1)] # (file name, label, number of frames)
    vid_loader = Video([['mk_video/fast_lap_01.mp4', 0], ['mk_video/slow_lap_01.mp4', 1]])
    train_data_iter, test_data_iter = vid_loader.start_here()

    mk_net = MKnet()
    smce_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # trainer = gluon.Trainer(mk_net.collect_params(), 'sgd', {'learning_rate': 0.001, 'wd': 1e-6})
    trainer = gluon.Trainer(mk_net.collect_params(), 'sgd', {'learning_rate': 0.001})

    train(mk_net, train_data_iter, test_data_iter, 100, trainer, smce_loss)

if __name__ == '__main__':
    main()