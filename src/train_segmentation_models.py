from SegmentationModel import SegmentationModel

''' Substitute the label name of the majority class for which the network has to be trained '''


net = SegmentationModel('dog')
net.train(epochs=300)
