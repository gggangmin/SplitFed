### # SplitFed
Hierarchical Federated Learning with model split

### environment
based on Flower, Pytorch

### Desc
The structure of the system consists of cloud server, edge server, and edge device. The edge server and the edge device share a model and update it. The model of the edge device is processed in parallel by several edge devices. The feature generated from the edge device is collected in the edge server and further learned. The data collected in the edge server is sequentially learned. The edge server has a model of the edge device, making it easy to back propagation. The edge server performs federated learning with the cloud server.
