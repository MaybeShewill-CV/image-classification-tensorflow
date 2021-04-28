// Copyright 2021 MaybeShewill-CV All Rights Reserved.
// Author: MaybeShewill-CV
// File: wf_net_training_monitor.cpp
// Date: 2021/04/26 下午13:45

// network training process monitor service main function

#include <iostream>

#include <workflow/WFHttpServer.h>

#include "utils/monitor_utils.h"

int main() {

    std::string checkpoint_model_save_dir;
    wf_monitor::utils::get_checkpoint_model_save_dir("/home/baidu/Silly_Project/ICode/baidu/beec/image-classification-tensorflow", checkpoint_model_save_dir);
    LOG(INFO) << checkpoint_model_save_dir;

    std::string checkpoint_model_path;
    wf_monitor::utils::get_latest_checkpoint(checkpoint_model_save_dir, checkpoint_model_path);
    LOG(INFO) << checkpoint_model_path;

    std::string dataset_name;
    std::string dataset_flag;
    int image_count;
    float precision;
    float recall;
    float f1;
    wf_monitor::utils::get_checkpoint_model_eval_statics(
            "/home/baidu/Silly_Project/ICode/baidu/beec/image-classification-tensorflow",
            dataset_name, dataset_flag, &image_count, &precision, &recall, &f1);
    LOG(INFO) << dataset_name;
    LOG(INFO) << dataset_flag;
    LOG(INFO) << image_count;
    LOG(INFO) << precision;
    LOG(INFO) << recall;
    LOG(INFO) << f1;

    WFHttpServer server([](WFHttpTask *task) {
        task->get_resp()->append_output_body("<html>Hello World!</html>");
    });

    if (server.start(8888) == 0) { // start server on port 8888
        getchar(); // press "Enter" to end.
        server.stop();
    }
    return 0;
}
