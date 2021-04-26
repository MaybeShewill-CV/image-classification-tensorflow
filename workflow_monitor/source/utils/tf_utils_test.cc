// Copyright 2021 MaybeShewill-CV All Rights Reserved..
// Author: MaybeShewill-CV
// File: tf_utils_test.cc.c
// Date: 2021/4/26 下午2:35

// tf utils test

#include "gtest/gtest.h"

#include "tf_utils.h"

using namespace wf_monitor::utils;

class TFUtilsTest : public testing::Test {
protected:
    void SetUp() override {
        puts("SetUp()");
    }
    void TearDown() override {
        puts("TearDown()");
    }
};

TEST_F(TFUtilsTest, Test_Getlatestcheckpoint) {
    std::string model_dir = "/home/baidu/Silly_Project/ICode/"
                            "baidu/beec/image-classification-tensorflow/model/resnet_ilsvrc_2012";
    std::string checkpoint_model_path;
    std::string checkpoint_model_path_gt = "/home/baidu/Silly_Project/ICode/baidu/beec/"
                                           "image-classification-tensorflow/model/"
                                           "resnet_ilsvrc_2012/resnet_val_acc=0.3295.ckpt-8";
    ASSERT_EQ(get_latest_checkpoint("", checkpoint_model_path), false);
    ASSERT_EQ(get_latest_checkpoint(model_dir, checkpoint_model_path), true);
    ASSERT_EQ(checkpoint_model_path, checkpoint_model_path_gt);
}
