// Copyright 2021 MaybeShewill-CV All Rights Reserved.
// Author: MaybeShewill-CV
// File: wf_net_training_monitor.cpp
// Date: 2021/04/26 下午13:45

// network training process monitor service main function

#include <iostream>

#include <workflow/WFHttpServer.h>
#include <toml/toml.hpp>
#include <glog/logging.h>

#include "server/model_status_monitor_server.h"

int main(int argc, char** argv) {

    wf_monitor::server::model_stat_monitor_server::ModelStatusMonitorServer server;
    if (server.start(std::atoi(argv[1])) == 0) {
        server.wait_finish();
    } else {
        perror("start server");
        exit(1);
    }

    return 0;
}
