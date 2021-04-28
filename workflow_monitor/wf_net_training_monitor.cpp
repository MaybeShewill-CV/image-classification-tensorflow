// Copyright 2021 MaybeShewill-CV All Rights Reserved.
// Author: MaybeShewill-CV
// File: wf_net_training_monitor.cpp
// Date: 2021/04/26 下午13:45

// network training process monitor service main function

#include <iostream>

#include <workflow/WFHttpServer.h>

#include "source/utils/monitor_utils.h"

int main() {

    std::string checkpoint_model_save_dir;
    wf_monitor::utils::get_checkpoint_model_save_dir("/home/baidu/Silly_Project/ICode/baidu/beec/image-classification-tensorflow", checkpoint_model_save_dir);
    LOG(INFO) << checkpoint_model_save_dir;

    std::string checkpoint_model_path;
    wf_monitor::utils::get_latest_checkpoint(checkpoint_model_save_dir, checkpoint_model_path);
    LOG(INFO) << checkpoint_model_path;

    int epoch = 0;
    float train_loss = 0.0f;
    float test_loss = 0.0f;
    float train_acc = 0.0f;
    float test_acc = 0.0f;
    wf_monitor::utils::get_model_training_statics("/home/baidu/Silly_Project/ICode/baidu/beec/image-classification-tensorflow", &epoch, &train_loss, &test_loss, &train_acc, &test_acc);

    WFHttpServer server([](WFHttpTask *task) {
        task->get_resp()->append_output_body("<html>Hello World!</html>");
    });

    if (server.start(8888) == 0) { // start server on port 8888
        getchar(); // press "Enter" to end.
        server.stop();
    }
    return 0;
}
