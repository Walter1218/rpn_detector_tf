# multi_scale_detector

1. the basical detector model was public and the usage of target label generate can see rpn_target_label_generate.ipynb, the usage of inference code please see rpn_inference.ipynb (I reorgnize the Faster RCNN code, and this is different tensorflow version with other github tf version), for FDDB dataset please see another two ipynb file.

2. multi_scale detector will support soon

3. This version now support FDDB, VOC dataset, I will add KITTI support soon

4. The key point regression part will add in next version(Because of the few dataset, the model works not good, add later).

4. The face recognition model will use Hinge Loss, Softmax loss and Center Loss to train.

5. for small obj, the multi feature based region proposal model can easily detector, but because of the receptive field, it become low resolution, so for small object classification, we need add a network can have low resolution input to high resolution output. This network will also add soon (using perceptual loss or GAN based network structure).

6. Add mAP evalution code

7. add fast part(proposal nms & target proposaled label) #TODO

The results of using RPN based network as face detector please see in ./example/

Author:Walter

Email:walter.1218@hotmail.com
