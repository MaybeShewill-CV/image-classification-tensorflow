// Copyright 2021 MaybeShewill-CV All Rights Reserved.
// Author: MaybeShewill-CV
// File: project_monitor.cpp
// Date: 2021/04/29 上午1:47

#include "project_monitor.h"

#include "file_system_utils/file_system_processor.h"

#include "utils/monitor_utils.h"

namespace wf_monitor {
namespace project {

using beec::common::file_system_utils::FileSystemProcessor;
namespace wfm_utils = wf_monitor::utils;

/*********** Public Function Sets **********************/

ProjectMonitor::ProjectMonitor(const toml::value& config) {
    if (!config.contains("Image_Classification_Tensorflow")) {
        LOG(ERROR) << "Config file was not complete, please check again";
        _m_successfully_init = false;
        return;
    }
    toml::value project_cfg_content = config.at("Image_Classification_Tensorflow");
    if (!project_cfg_content.contains("project_base_dir")) {
        LOG(ERROR) << "Config file doesn\'t have project_base_dir field";
        _m_successfully_init = false;
        return;
    } else {
        _m_project_dir = project_cfg_content.at("project_base_dir").as_string();
    }
    if (!FileSystemProcessor::is_directory_exist(_m_project_dir)) {
        LOG(ERROR) << "Project base dir: " << _m_project_dir << ", not exist";
        _m_successfully_init = false;
        return;
    }
    _m_log_dir = FileSystemProcessor::combine_path(_m_project_dir, "log");
    if (!FileSystemProcessor::is_directory_exist(_m_log_dir)) {
        LOG(ERROR) << "Project log dir: " << _m_log_dir << ", not exist";
        _m_successfully_init = false;
        return;
    }
    _m_successfully_init = true;
}

/***
 *
 * @param model_name
 * @return
 */
bool ProjectMonitor::get_current_training_model_name(std::string &model_name) {
    return wfm_utils::MonitorUtils::get_training_model_name(_m_log_dir, model_name);
}

/***
 *
 * @param dataset_name
 * @return
 */
bool ProjectMonitor::get_current_training_dataset_name(std::string &dataset_name) {
    return wfm_utils::MonitorUtils::get_training_dataset_name(_m_log_dir, dataset_name);
}

/***
 *
 * @param stat
 * @return
 */
bool ProjectMonitor::get_latest_training_statics(TrainStatic &stat) {
    int epoch = 0;
    float train_loss = 0.0;
    float test_loss = 0.0;
    float train_acc = 0.0;
    float test_acc = 0.0;
    if (!wfm_utils::MonitorUtils::get_model_training_statics(
                _m_project_dir, &epoch, &train_loss, &test_loss, &train_acc, &test_acc)) {
        return false;
    }
    stat.epoch = epoch;
    stat.training_loss = train_loss;
    stat.testing_loss = test_loss;
    stat.training_accuracy = train_acc;
    stat.testing_accuracy = test_acc;
    return true;
}

/***
 *
 * @param stat
 * @return
 */
bool ProjectMonitor::get_latest_eval_statics(EvalStatic &stat) {
    std::string dataset_flag;
    std::string dataset_name;
    int image_count = 0;
    float precision = 0.0;
    float recall = 0.0;
    float f1 = 0.0;
    if (!wfm_utils::MonitorUtils::get_latest_checkpoint_model_eval_statics(
                _m_project_dir, dataset_name, dataset_flag, &image_count, &precision, &recall, &f1)) {
        return false;
    }
    stat.dataset_name = dataset_name;
    stat.dataset_flag = dataset_flag;
    stat.image_count = image_count;
    stat.precision = precision;
    stat.recall = recall;
    stat.f1 = f1;
    return true;
}

/***
 *
 * @param checkpoint_model_name
 * @return
 */
bool ProjectMonitor::is_checkpoint_model_evaluated(const std::string &checkpoint_model_name) {
    std::string eval_log_file_path;
    if (!wfm_utils::MonitorUtils::get_eval_log_file_path(_m_log_dir, eval_log_file_path)) {
        return false;
    }
    return wfm_utils::MonitorUtils::is_checkpoint_model_evaluated(eval_log_file_path, checkpoint_model_name);
}

/*********** Private Function Sets **********************/
}
}
