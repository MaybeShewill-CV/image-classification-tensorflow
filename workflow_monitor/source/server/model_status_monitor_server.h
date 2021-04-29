/************************************************
* Copyright 2021 MaybeShewill-CV All Rights Reserved..
* Author: MaybeShewill-CV
* File: modelStatusMonitorServer.h
* Date: 2021/4/29 下午3:49
************************************************/

#ifndef WORKFLOW_MONITOR_MODELSTATUSMONITORSERVER_H
#define WORKFLOW_MONITOR_MODELSTATUSMONITORSERVER_H

#include "workflow/WFHttpServer.h"
#include "toml/toml.hpp"

#include "project/project_monitor.h"

namespace wf_monitor {
namespace server {

namespace model_stat_monitor_server {
class ModelStatusMonitorServer {
public:
    /***
     *
     */
    ModelStatusMonitorServer();

    /***
     *
     */
    ~ModelStatusMonitorServer();


    /***
    * copy constructor
    * @param transformer
    */
    ModelStatusMonitorServer(const ModelStatusMonitorServer& transformer) = delete;

    /***
     * assign constructor
     * @param transformer
     * @return
     */
    ModelStatusMonitorServer &operator=(const ModelStatusMonitorServer& transformer) = delete;

    /***
    *
    * @param port
    * @return
    */
    int start(unsigned short port);

    /***
     *
     * @param host
     * @param port
     * @return
     */
    int start(const char *host, unsigned short port);

    /***
     *
     */
    void stop();

    /***
     *
     */
    void shutdown() {
        _m_server->shutdown();
    };

    /***
     *
     */
    void wait_finish() {
        _m_server->wait_finish();
    }

private:
    // http server
    WFHttpServer* _m_server = nullptr;
};
}
}
}

#endif //WORKFLOW_MONITOR_MODELSTATUSMONITORSERVER_H
