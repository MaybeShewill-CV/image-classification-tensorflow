/************************************************
* Copyright 2021 MaybeShewill-CV All Rights Reserved..
* Author: MaybeShewill-CV
* File: modelStatusMonitorServer.cpp.cc
* Date: 2021/4/29 下午3:49
************************************************/

#include "model_status_monitor_server.h"

#include <glog/logging.h>

#include "project/project_monitor.h"

namespace wf_monitor {
namespace server {

namespace model_stat_monitor_impl {

using wf_monitor::project::ProjectMonitor;
using InterfaceMap = std::map<std::string, std::function<std::string (WFHttpTask*)>>;

static ProjectMonitor* get_proj_monitor() {
    auto config = toml::parse("../conf/server_conf.ini");
    static ProjectMonitor *monitor = nullptr;
    if (monitor == nullptr) {
        monitor = new ProjectMonitor(config);
    } else if (!monitor->is_successfully_initialized()) {
        ProjectMonitor* tmp = monitor;
        delete tmp;
        tmp = nullptr;
        monitor = new ProjectMonitor(config);
    }
    return monitor;
}

std::string process_get_cur_train_model_name(void*) {
    auto* proj_monitor = get_proj_monitor();
    std::string cur_model_name;
    char buff[128];
    if (!proj_monitor->get_current_training_model_name(cur_model_name)) {
        LOG(INFO) << "get cur train model name failed";
    }
    sprintf(buff, "{\"model_name\": %s}", cur_model_name.c_str());
    return std::string(buff);
}

std::string process_get_cur_train_dataset_name(void*) {
    auto* proj_monitor = get_proj_monitor();
    std::string cur_dataset_name;
    char buff[128];
    if (!proj_monitor->get_current_training_dataset_name(cur_dataset_name)) {
        LOG(INFO) << "get cur train dataset name failed";
    }
    sprintf(buff, "{\"dataset_name\": %s}", cur_dataset_name.c_str());
    return std::string(buff);
}

std::string process_get_latest_train_statics(void*) {
    auto* proj_monitor = get_proj_monitor();
    wf_monitor::project::TrainStatic train_stat;
    if (!proj_monitor->get_latest_training_statics(train_stat)) {
        LOG(INFO) << "get latest train statics failed";
    }
    return train_stat.to_str();
}

std::string process_get_latest_eval_statics(void*) {
    auto* proj_monitor = get_proj_monitor();
    wf_monitor::project::EvalStatic eval_stat;
    if (!proj_monitor->get_latest_eval_statics(eval_stat)) {
        LOG(INFO) << "get latest eval statics failed";
    }
    return eval_stat.to_str();
}

std::string process_is_latest_checkpoint_model_evaluated(void*) {
    auto* proj_monitor = get_proj_monitor();
    if (!proj_monitor->is_latest_checkpoint_model_evaluated()) {
        return "{\"is_evaluated\": false}";
    } else {
        return "{\"is_evaluated\": true}";
    }
}

std::string process_get_latest_checkpoint_model_path(void*) {
    auto* proj_monitor = get_proj_monitor();
    std::string checkpoint_model_path;
    char buff[256];
    if (!proj_monitor->get_latest_checkpoint_model_path(checkpoint_model_path)) {
        LOG(INFO) << "Get latest checkpoint model path failed";
    }
    sprintf(buff, "{\"checkpoint_model_path\": %s}", checkpoint_model_path.c_str());
    return std::string(buff);
}

std::string process_get_current_train_epoch(void*) {
    auto* proj_monitor = get_proj_monitor();
    int epoch = 0;
    char buff[128];
    if (!proj_monitor->get_current_train_epoch(&epoch)) {
        LOG(INFO) << "Get current train epoch failed";
    }
    sprintf(buff, "{\"epoch\": %d}", epoch);
    return std::string(buff);
}

std::string process_is_training_process_alive(void*) {
    if (!wf_monitor::project::ProjectMonitor::is_training_process_alive()) {
        return "{\"is_alive\": false}";
    } else {
        return "{\"is_alive\": true}";
    }
}

std::string process_is_evaluating_process_alive(void*) {
    if (!wf_monitor::project::ProjectMonitor::is_evaluating_process_alive()) {
        return "{\"is_alive\": false}";
    } else {
        return "{\"is_alive\": true}";
    }
}

void _apply_eval_on_latest_checkpoint_model(std::string *project_dir, std::string *output) {
    auto* proj_monitor = get_proj_monitor();
    std::string model_name;
    std::string dataset_name;
    std::string checkpoint_model_path;
    if (!proj_monitor->get_current_training_model_name(model_name) ||
    !proj_monitor->get_current_training_dataset_name(dataset_name) ||
    !proj_monitor->get_latest_checkpoint_model_path(checkpoint_model_path)) {
        LOG(ERROR) << "Fetch eval scripts params failed";
        return;
    }

    char command_buf[512];
//    sprintf(command_buf, "nohup bash %s/scripts/evaluate_model.sh %s %s %s %s > out.file 2>&1 &",
//            project_dir->c_str(), model_name.c_str(), dataset_name.c_str(),
//            checkpoint_model_path.c_str(), project_dir->c_str());
    sprintf(command_buf, "bash %s/scripts/evaluate_model.sh %s %s %s %s",
            project_dir->c_str(), model_name.c_str(), dataset_name.c_str(),
            checkpoint_model_path.c_str(), project_dir->c_str());
    LOG(INFO) << "Eval command: " << command_buf;
    FILE* fp = nullptr;
    if ((fp = popen(command_buf, "r")) == nullptr) {
        LOG(ERROR) << "popen err";
        return;
    }
    pclose(fp);
    fp = nullptr;
}

std::string process_auto_eval_latest_checkpoint_model(WFHttpTask* task) {
    if (wf_monitor::project::ProjectMonitor::is_evaluating_process_alive()) {
        return R"({"status": -1, "msg": an evaluating process was alive})";
    }
    auto* proj_monitor = get_proj_monitor();
    if (proj_monitor->is_latest_checkpoint_model_evaluated()) {
        return R"({"status": -1, "msg": latest checkpoint model has been evaluated})";
    }

    std::string proj_base_dir;
    proj_monitor->get_project_base_dir(proj_base_dir);
    std::string model_name;
    std::string dataset_name;
    std::string checkpoint_model_path;
    if (!proj_monitor->get_current_training_model_name(model_name) ||
        !proj_monitor->get_current_training_dataset_name(dataset_name) ||
        !proj_monitor->get_latest_checkpoint_model_path(checkpoint_model_path)) {
        return R"({"status": -1, "msg": fetch eval scripts params failed})";
    }

    char command_buf[512];
    sprintf(command_buf, "nohup bash %s/scripts/evaluate_model.sh %s %s %s %s > out.file 2>&1 &",
            proj_base_dir.c_str(), model_name.c_str(), dataset_name.c_str(),
            checkpoint_model_path.c_str(), proj_base_dir.c_str());
//    sprintf(command_buf, "bash %s/scripts/evaluate_model.sh %s %s %s %s",
//            proj_base_dir.c_str(), model_name.c_str(), dataset_name.c_str(),
//            checkpoint_model_path.c_str(), proj_base_dir.c_str());
    LOG(INFO) << "Eval command: " << command_buf;
    FILE* fp = nullptr;
    if ((fp = popen(command_buf, "r")) == nullptr) {
        LOG(ERROR) << "popen err";
        return R"({"status": -1, "msg": popen run eval scripts failed})";;
    }
    pclose(fp);
    fp = nullptr;
}


static InterfaceMap* init_interface_map() {
    static InterfaceMap *interface_map = nullptr;
    if (interface_map == nullptr) {
        interface_map = new InterfaceMap();
        // register interface function
        interface_map->insert(
            std::make_pair("/get_cur_train_model_name", process_get_cur_train_model_name));
        interface_map->insert(
            std::make_pair("/get_cur_train_dataset_name", process_get_cur_train_dataset_name));
        interface_map->insert(
            std::make_pair("/get_latest_train_statics", process_get_latest_train_statics));
        interface_map->insert(
            std::make_pair("/get_latest_eval_statics", process_get_latest_eval_statics));
        interface_map->insert(
                std::make_pair("/get_latest_checkpoint_path", process_get_latest_checkpoint_model_path));
        interface_map->insert(
                std::make_pair("/get_current_train_epoch", process_get_current_train_epoch));
        interface_map->insert(
                std::make_pair("/is_latest_checkpoint_model_evaluated",
                               process_is_latest_checkpoint_model_evaluated));
        interface_map->insert(
                std::make_pair("/is_training_process_alive", process_is_training_process_alive));
        interface_map->insert(
                std::make_pair("/is_evaluating_process_alive", process_is_evaluating_process_alive));
        interface_map->insert(
                std::make_pair("/auto_eval_latest_checkpoint_model", process_auto_eval_latest_checkpoint_model));
    }
    return interface_map;
}

void server_process(WFHttpServer *server, WFHttpTask *task) {
    auto uri = task->get_req()->get_request_uri();
    auto* interface_func = init_interface_map();

    if (strcmp(task->get_req()->get_request_uri(), "/stop") == 0) {
        DLOG(INFO) << "Request-URI: " << task->get_req()->get_request_uri();
        static std::atomic<int> flag;
        if (flag++ == 0) {
            server->shutdown();
        }
        task->get_resp()->append_output_body("<html>Model status monitor server stop</html>");
        return;
    }
    if (interface_func->find(uri) == interface_func->end()) {
        task->get_resp()->append_output_body("<html>No such api registered</html>");
        return;
    } else {
        auto proc_func = interface_func->find(uri)->second;
        auto response_body = proc_func(task);
        task->get_resp()->append_output_body(response_body + '\n');
        return;
    }
}
}

namespace model_stat_monitor_server {

/*****************Public Function Sets****************/

/***
 * 构造函数
 * @param config
 */
ModelStatusMonitorServer::ModelStatusMonitorServer() {
    struct WFServerParams params = HTTP_SERVER_PARAMS_DEFAULT;
    params.max_connections = 200;
    params.peer_response_timeout = 30 * 1000;
    params.ssl_accept_timeout = 30 * 1000;
    auto&& proc = std::bind(
                      model_stat_monitor_impl::server_process,
                      std::cref(this->_m_server),
                      std::placeholders::_1);
    _m_server = new WFHttpServer(&params, proc);
}

/***
 * 析构函数
 */
ModelStatusMonitorServer::~ModelStatusMonitorServer() {
    if (_m_server != nullptr) {
        delete _m_server;
        _m_server = nullptr;
    }
    if (model_stat_monitor_impl::init_interface_map() != nullptr) {
        delete model_stat_monitor_impl::init_interface_map();
    }
    if (model_stat_monitor_impl::get_proj_monitor() != nullptr) {
        delete model_stat_monitor_impl::get_proj_monitor();
        LOG(INFO) << "Release project monitor";
    }
    LOG(INFO) << "Quit model status monitor server";
}

/***
 * 启动函数
 * @param port
 * @return
 */
int ModelStatusMonitorServer::start(unsigned short port) {
    return _m_server->start(port);
}

/***
 *
 * @param host
 * @param port
 * @return
 */
int ModelStatusMonitorServer::start(const char *host, unsigned short port) {
    return _m_server->start(host, port);
}

/***
 *
 */
void ModelStatusMonitorServer::stop() {
    return _m_server->stop();
}
}
}
}
