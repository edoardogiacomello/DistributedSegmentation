from SegmentationModel import SegmentationModel

net = SegmentationModel('dog')
net.train(epochs=300)
