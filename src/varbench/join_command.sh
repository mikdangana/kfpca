To add a worker node, run the following command on the worker node:
curl -sfL https://get.k3s.io | K3S_URL=https://192.168.41.232:6443 K3S_TOKEN=K10f6922390533313e899f34b576794ade6e665f1fdcb6a5b5befe3d8898f078bb6::server:132d906fc305f16af31358f100e63f7d sh -s - --with-node-id
