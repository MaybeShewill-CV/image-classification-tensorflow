// Copyright 2021 MaybeShewill-CV All Rights Reserved.
// Author: MaybeShewill-CV
// File: wf_net_training_monitor.cpp
// Date: 2021/04/26 下午13:45

// network training process monitor service main function

#include <iostream>

#include <workflow/WFHttpServer.h>

#include "source/utils/monitor_utils.h"

int main() {

    while (true) {
        std::string l_path;
        wf_monitor::utils::get_latested_training_log_file(
                "/home/baidu/Silly_Project/ICode/baidu/beec/image-classification-tensorflow/log",
                l_path
        );
        LOG(INFO) << l_path;

    }

    WFHttpServer server([](WFHttpTask *task) {
        task->get_resp()->append_output_body("<html>Hello World!</html>");
    });

    if (server.start(8888) == 0) { // start server on port 8888
        getchar(); // press "Enter" to end.
        server.stop();
    }
    return 0;
}
