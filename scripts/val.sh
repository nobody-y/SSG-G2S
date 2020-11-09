#!/usr/bin/env bash
export PYTHONPATH=/home/***/lib/apex:/home/***/lib/cocoapi:/home/***/code/scene_graph_gen/scene_graph_benchmark_pytorch:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=3,4
    export NUM_GUP=2
    echo "Testing Predcls"
    MODEL_NAME="transformer_predcls_dist15_2k_IniPModel_CleanHead_Lr1e3_B16"
    for j in {1..3}
    do     
    for k in {1..3}
    do 
    for h in {1..3}
    do    
        echo "VAL_ALPHA"
        echo $j
        alpha_j=$(echo "0.3 * $j"|bc)
        echo $alpha_j
        echo $k
        alpha_k=$(echo "0.3 * $k"|bc)
        echo $alpha_k
        echo $h
        alpha_h=$(echo "0.3 * $h"|bc)
        echo $alpha_h
        python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GUP \
                tools/relation_test_net.py \
                --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
                MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
                MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
                MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerSuperPredictor \
                TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
                GLOVE_DIR ./datasets/vg/ \
                MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
                MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0004000.pth \
                OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
                TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG True \
                MODEL.ROI_RELATION_HEAD.VAL_ALPHA_10 $alpha_j \
                MODEL.ROI_RELATION_HEAD.VAL_ALPHA_15 $alpha_k \
                MODEL.ROI_RELATION_HEAD.VAL_ALPHA_20 $alpha_h;
    done
    done
    done
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GUP=1
    echo "Testing Predcls"
    MODEL_NAME="transformer_predcls_dist15_2k_NoGradCClassifier_InitPreModel_lr1e4"
for i in {0..2}
do
    echo "TRAINING MOTIFNET"
    echo $i
    alpha_i=$(echo "0.1 * $i"|bc)
    echo $alpha_i
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerSuperNoGradPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA $alpha_i;
            #TEST.ALLOW_LOAD_FROM_CACHE True
done
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "Testing SGCls"
    python  -u -m torch.distributed.launch --nproc_per_node=2 \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
            TEST.IMS_PER_BATCH 1 DTYPE "float16" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/transformer_super_sgcls/model_0012000.pth \
            OUTPUT_DIR ./checkpoints/transformer_super_sgcls
elif [ $1 == "3" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "Testing SGDet"
    python  -m torch.distributed.launch --nproc_per_node=2 \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
            TEST.IMS_PER_BATCH 1 DTYPE "float16" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/transformer_super_sgdet/model_0012000.pth \
            OUTPUT_DIR ./checkpoints/transformer_super_sgdet
fi
