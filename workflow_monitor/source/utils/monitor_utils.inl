// Copyright 2021 MaybeShewill-CV All Rights Reserved.
// Author: MaybeShewill-CV
// File: monitor_utils.inl
// Date: 2021/4/29 上午11:03

#include <fstream>
#include <algorithm>
#include <sys/stat.h>

#include <glog/logging.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include "file_system_utils//file_system_processor.h"
#include "project/project_monitor.h"

namespace wf_monitor {
namespace utils {

using beec::common::file_system_utils::FileSystemProcessor;
namespace fs = boost::filesystem;

/************** Public Function Sets **************/

inline bool MonitorUtils::get_latest_checkpoint_path(const std::string &model_dir,
        std::string &model_checkpoint_path) {
    if (!FileSystemProcessor::is_directory_exist(model_dir)) {
        LOG(ERROR) << "Model dir: " << model_dir << ", not exist";
        return false;
    }
    // read checkpoint file
    std::string check_point_record_file_name = "checkpoint";
    std::string check_point_record_file_path = FileSystemProcessor::combine_path(
                model_dir, check_point_record_file_name);
    if (!FileSystemProcessor::is_file_exist(check_point_record_file_path)) {
        LOG(ERROR) << "Check point file path: " << check_point_record_file_path << ", not exist";
        return false;
    }
    // read lasted checkpoint model name from checkpoint file
    std::ifstream checkpoint_file;
    checkpoint_file.open(check_point_record_file_path, std::fstream::in);
    if (!checkpoint_file.is_open() || !checkpoint_file.good()) {
        LOG(ERROR) << "Open checkpoint file: " << check_point_record_file_path << " failed";
        return false;
    }
    const int char_size = 128;
    char checkpoint_model_name_cstr[char_size];
    checkpoint_file.getline(checkpoint_model_name_cstr, char_size);
    std::string checkpoint_model_name(checkpoint_model_name_cstr);
    checkpoint_model_name = checkpoint_model_name.substr(
            checkpoint_model_name.find_first_of('\"') + 1,
            checkpoint_model_name.find_last_of('\"') - checkpoint_model_name.find_first_of('\"') - 1
            );
    // check index file path
    std::string index_file_name = checkpoint_model_name + ".index";
    std::string index_file_path = FileSystemProcessor::combine_path(model_dir, index_file_name);
    if (!FileSystemProcessor::is_file_exist(index_file_path)) {
        LOG(ERROR) << "Index file: " << index_file_path << ", not exist";
        return false;
    }
    // check meta file path
    std::string meta_file_name = checkpoint_model_name + ".meta";
    std::string meta_file_path = FileSystemProcessor::combine_path(model_dir, meta_file_name);
    if (!FileSystemProcessor::is_file_exist(meta_file_path)) {
        LOG(ERROR) << "Meta file: " << meta_file_path << ", not exist";
        return false;
    }
    // check data file path
    std::string data_file_name = checkpoint_model_name + ".data-00000-of-00001";
    std::string data_file_path = FileSystemProcessor::combine_path(model_dir, data_file_name);
    if (!FileSystemProcessor::is_file_exist(data_file_path)) {
        LOG(ERROR) << "Data file: " << data_file_path << ", not exist";
        return false;
    }
    checkpoint_file.close();

    model_checkpoint_path = FileSystemProcessor::combine_path(model_dir, checkpoint_model_name);
    return true;
}

inline bool MonitorUtils::is_checkpoint_model_evaluated(const std::string &eval_log_file_path,
        const std::string &model_name) {
    std::ifstream eval_file;
    eval_file.open(eval_log_file_path, std::fstream::in);
    if (!eval_file.is_open() || !eval_file.good()) {
        LOG(ERROR) << "Open evaluation record file: " << eval_log_file_path << ", failed";
        return false;
    }

    std::string record_info;
    bool model_has_been_evaluated = false;
    while (std::getline(eval_file, record_info)) {
        if (record_info.find(model_name) != std::string::npos) {
            model_has_been_evaluated = true;
            break;
        }
    }
    eval_file.close();
    return model_has_been_evaluated;
}

inline bool MonitorUtils::is_net_training_process_alive() {
    return _is_process_alive("train_model.py");
}

inline bool MonitorUtils::is_net_evaluating_process_alive() {
    return _is_process_alive("evaluate_model.py");
}

inline bool MonitorUtils::get_all_training_log_files(const std::string &log_dir,
        std::vector<std::string> &training_log_files) {
    if (!FileSystemProcessor::is_directory_exist(log_dir)) {
        return false;
    }
    std::vector<std::string> tmp_log_files;
    FileSystemProcessor::get_directory_files(
        log_dir, tmp_log_files, ".log",
        FileSystemProcessor::SEARCH_OPTION_T::TOPDIRECTORYONLY);
    training_log_files.clear();
    for (const auto& file_path : tmp_log_files) {
        auto file_name = FileSystemProcessor::get_file_name(file_path);
        if (file_name.find("classification_train") != std::string::npos) {
            training_log_files.push_back(file_path);
        }
    }
    return !training_log_files.empty();
}

inline std::time_t MonitorUtils::get_file_last_modified_time(const std::string &file_path) {
    if (!FileSystemProcessor::is_file_exist(file_path)) {
        return 0;
    }
    struct stat buf{};
    FILE *pFile = nullptr;
    pFile = fopen(file_path.c_str(), "r");
    int fd = fileno(pFile);
    fstat(fd, &buf);
    std::time_t time = buf.st_mtime;
    return time;
}

inline bool MonitorUtils::get_latest_training_log_file(const std::string &log_dir,
        std::string &latested_log_file_path) {
    std::vector<std::string> tmp_log_file_paths;
    if (!get_all_training_log_files(log_dir, tmp_log_file_paths)) {
        latested_log_file_path = "";
        return false;
    }
    std::map<std::time_t, std::string> tmp_log_file_map;
    for (const auto& tmp_file_path : tmp_log_file_paths) {
        tmp_log_file_map.insert(std::make_pair(get_file_last_modified_time(tmp_file_path), tmp_file_path));
    }
    latested_log_file_path = tmp_log_file_map.rbegin()->second;
    return true;
}

inline bool MonitorUtils::get_training_model_name(const std::string &log_dir, std::string &model_name) {
    if (!is_net_training_process_alive()) {
        LOG(INFO) << "Get training model name failed";
        model_name = "";
        return false;
    }

    std::string latest_log_file_path;
    if (!get_latest_training_log_file(log_dir, latest_log_file_path)) {
        LOG(INFO) << "Get training model name failed";
        model_name = "";
        return false;
    }
    std::string latest_log_file_name = FileSystemProcessor::get_file_name(latest_log_file_path);
    std::string dataset_model_name = latest_log_file_name.substr(
            0, latest_log_file_name.find("classification") - 1);
    model_name = dataset_model_name.substr(dataset_model_name.find_last_of('_') + 1);
    return true;
}

inline bool MonitorUtils::get_training_dataset_name(const std::string &log_dir, std::string &dataset_name) {
    if (!is_net_training_process_alive()) {
        LOG(INFO) << "Get training model name failed";
        dataset_name = "";
        return false;
    }

    std::string latest_log_file_path;
    if (!get_latest_training_log_file(log_dir, latest_log_file_path)) {
        LOG(INFO) << "Get training model name failed";
        dataset_name = "";
        return false;
    }
    std::string latest_log_file_name = FileSystemProcessor::get_file_name(latest_log_file_path);
    std::string dataset_model_name = latest_log_file_name.substr(
                                         0, latest_log_file_name.find("classification") - 1);
    dataset_name = dataset_model_name.substr(0, dataset_model_name.find_last_of('_'));
    return true;
}

inline bool MonitorUtils::get_checkpoint_model_save_dir(const std::string &project_base_dir,
        std::string &model_save_dir) {
    std::string project_log_dir = FileSystemProcessor::combine_path(project_base_dir, "log");
    std::string model_name;
    if (!get_training_model_name(project_log_dir, model_name)) {
        LOG(INFO) << "Get model save dir failed";
        model_save_dir = "";
        return false;
    }

    std::string dataset_name;
    if (!get_training_dataset_name(project_log_dir, dataset_name)) {
        LOG(INFO) << "Get model save dir failed";
        model_save_dir = "";
        return false;
    }

    char model_save_dir_name[256];
    sprintf(model_save_dir_name, "%s_%s", model_name.c_str(), dataset_name.c_str());
    std::string model_root_dir = FileSystemProcessor::combine_path(project_base_dir, "model");
    model_save_dir = FileSystemProcessor::combine_path(model_root_dir, std::string(model_save_dir_name));
    if (!FileSystemProcessor::is_directory_exist(model_save_dir)) {
        LOG(ERROR) << "Model save dir: " << model_save_dir << ", not exist";
        model_save_dir = "";
        return false;
    }
    return true;
}

inline bool MonitorUtils::get_eval_log_file_path(const std::string &log_dir, std::string &eval_log_file_path) {
    std::string model_name;
    std::string dataset_name;
    if (!get_training_model_name(log_dir, model_name) ||
            !get_training_dataset_name(log_dir, dataset_name)) {
        LOG(ERROR) << "Get eval log file path failed";
        eval_log_file_path = "";
        return false;
    }

    char eval_log_file_name[128];
    sprintf(eval_log_file_name, "%s_%s_evaluate.log", dataset_name.c_str(), model_name.c_str());
    eval_log_file_path = FileSystemProcessor::combine_path(log_dir, eval_log_file_name);
    if (!FileSystemProcessor::is_file_exist(eval_log_file_path)) {
        LOG(ERROR) << "Eval log file path: " << eval_log_file_path << ", not exist";
        eval_log_file_path = "";
        return false;
    }
    return true;
}

inline bool MonitorUtils::get_latest_checkpoint_model_eval_statics(const std::string &project_dir,
        std::string &dataset_name,
        std::string &dataset_flag, int32_t *image_count,
        float_t *precision, float_t *recall, float_t *f1) {
    // get log file dir
    std::string training_log_dir = FileSystemProcessor::combine_path(project_dir, "log");
    if (!FileSystemProcessor::is_directory_exist(training_log_dir)) {
        LOG(ERROR) << "Training log dir: " << training_log_dir << ", not exist, get model eval statics failed";
        dataset_name = "";
        dataset_flag = "";
        *image_count = 0;
        *precision = 0.0;
        *recall = 0.0;
        *f1 = 0.0;
        return false;
    }
    // get eval log file path
    std::string eval_log_file_path;
    if (!get_eval_log_file_path(training_log_dir, eval_log_file_path)) {
        LOG(ERROR) << "Training log dir: " << training_log_dir << ", not exist, get model eval statics failed";
        dataset_name = "";
        dataset_flag = "";
        *image_count = 0;
        *precision = 0.0;
        *recall = 0.0;
        *f1 = 0.0;
        return false;
    }
    return _get_checkpoint_model_eval_statics_impl(
               eval_log_file_path, dataset_name,
               dataset_flag, image_count, precision, recall, f1);
}

inline bool MonitorUtils::get_model_training_statics(
        const std::string &project_dir, int *epoch, float *train_loss,
        float *test_loss, float *train_acc, float *test_acc) {
    std::string training_log_dir = FileSystemProcessor::combine_path(project_dir, "log");
    if (!FileSystemProcessor::is_directory_exist(training_log_dir)) {
        *epoch = 0;
        *train_loss = 0.0;
        *test_loss = 0.0;
        *train_acc = 0.0;
        *test_acc = 0.0;
        return false;
    }
    std::string train_log_file_path;
    if (!get_latest_training_log_file(training_log_dir, train_log_file_path)) {
        *epoch = 0;
        *train_loss = 0.0;
        *test_loss = 0.0;
        *train_acc = 0.0;
        *test_acc = 0.0;
        return false;
    }
    return _get_model_training_statics_impl(
               train_log_file_path, epoch, train_loss, test_loss, train_acc, test_acc);
}

inline bool MonitorUtils::get_cur_train_epoch(const std::string& project_dir, int* epoch) {

    std::string training_log_dir = FileSystemProcessor::combine_path(project_dir, "log");
    if (!FileSystemProcessor::is_directory_exist(training_log_dir)) {
        *epoch = 0;
        return false;
    }
    std::string train_log_file_path;
    if (!get_latest_training_log_file(training_log_dir, train_log_file_path)) {
        *epoch = 0;
        return false;
    }
    float train_loss = 0.0;
    float test_loss = 0.0;
    float train_acc = 0.0;
    float test_acc = 0.0;
    if (!_get_model_training_statics_impl(
            train_log_file_path, epoch, &train_loss, &test_loss, &train_acc, &test_acc)) {
        *epoch = 0;
        return false;
    }
    return true;
}

/************** Private Function Sets **************/

inline bool MonitorUtils::_get_checkpoint_model_eval_statics_impl(const std::string &eval_log_file_path,
        std::string &dataset_name,
        std::string &dataset_flag,
        int32_t *image_count, float_t *precision,
        float_t *recall, float_t *f1) {
    std::ifstream eval_file;
    eval_file.open(eval_log_file_path, std::fstream::in);
    if (!eval_file.is_open() || !eval_file.good()) {
        LOG(ERROR) << "Open evaluation record file: " << eval_log_file_path << ", failed";
        dataset_name = "";
        dataset_flag = "";
        *image_count = 0;
        *precision = 0.0;
        *recall = 0.0;
        *f1 = 0.0;
        return false;
    }

    std::map<int, wf_monitor::project::EvalStatic> eval_statics;

    std::string record_info;
    while (std::getline(eval_file, record_info)) {
        if (record_info.find("Eval model weights path:") != std::string::npos) {
            //
            auto checkpoint_name = record_info.substr(
                    record_info.find_last_of('/') + 1,
                    record_info.size() - record_info.find_last_of('/') - 1);
            auto epoch_nums = std::atoi(checkpoint_name.substr(checkpoint_name.find('-') + 1).c_str());

            wf_monitor::project::EvalStatic tmp_eval_stat;
            std::string tmp_info;
            // read dataset name
            std::getline(eval_file, tmp_info);
            tmp_eval_stat.dataset_name = tmp_info.substr(tmp_info.find_last_of(':') + 2);
            // read dataset flag
            std::getline(eval_file, tmp_info);
            tmp_eval_stat.dataset_flag = tmp_info.substr(tmp_info.find_last_of(':') + 2);
            // read dataset image count
            std::getline(eval_file, tmp_info);
            tmp_eval_stat.image_count = std::atoi(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
            // read model name
            std::getline(eval_file, tmp_info);
            std::string model_name = tmp_info.substr(tmp_info.find_last_of(':') + 1);
            // read model precision
            std::getline(eval_file, tmp_info);
            tmp_eval_stat.precision = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
            // read model recall
            std::getline(eval_file, tmp_info);
            tmp_eval_stat.recall = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
            // read model f1
            std::getline(eval_file, tmp_info);
            tmp_eval_stat.f1 = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
            eval_statics.insert(std::make_pair(epoch_nums, tmp_eval_stat));
        }
    }
    eval_file.close();
    auto latest_eval_stat = eval_statics.rbegin()->second;
    dataset_name = latest_eval_stat.dataset_name;
    dataset_flag = latest_eval_stat.dataset_flag;
    *image_count = latest_eval_stat.image_count;
    *precision = latest_eval_stat.precision;
    *recall = latest_eval_stat.recall;
    *f1 = latest_eval_stat.f1;
    return true;
}

inline bool MonitorUtils::_get_model_training_statics_impl(const std::string &trainning_log_file_path,
        int *epoch, float *train_loss, float *test_loss,
        float *train_acc, float *test_acc) {
    if (!FileSystemProcessor::is_file_exist(trainning_log_file_path)) {
        LOG(ERROR) << "Training log file: " << trainning_log_file_path << ", not exist";
        return false;
    }

    std::ifstream train_log_file;
    train_log_file.open(trainning_log_file_path, std::fstream::in);
    if (!train_log_file.is_open() || !train_log_file.good()) {
        LOG(ERROR) << "Open training log file: " << trainning_log_file_path << ", failed";
        return false;
    }

    train_log_file.seekg(-1, std::ios_base::end);
    if(train_log_file.peek() == '\n') {
        train_log_file.seekg(-1, std::ios_base::cur);
        int i = train_log_file.tellg();
        for (i; i > 0; i--) {
            if (train_log_file.peek() == '\n') {
                //Found
                train_log_file.get();
                break;
            }
            train_log_file.seekg(i, std::ios_base::beg);
        }
    }
    std::string latested_training_log_info;
    std::getline(train_log_file, latested_training_log_info);

    if (latested_training_log_info.find("INFO => Epoch:") == std::string::npos) {
        *epoch = 0;
        *train_loss = 0.0;
        *test_loss = 0.0;
        *train_acc = 0.0;
        *test_loss = 0.0;
        return false;
    }

    std::string epoch_flag = "Epoch:";
    std::string time_flag = "Time:";
    std::string train_loss_flag = "Train loss:";
    std::string test_loss_flag = "Test loss:";
    std::string train_acc_flag = "Train acc:";
    std::string test_acc_flag = "Test acc:";
    std::string end_flag = "...";

    auto epoch_idx = latested_training_log_info.find(epoch_flag);
    auto time_idx = latested_training_log_info.find(time_flag);
    auto train_loss_idx = latested_training_log_info.find(train_loss_flag);
    auto test_loss_idx = latested_training_log_info.find(test_loss_flag);
    auto train_acc_idx = latested_training_log_info.find(train_acc_flag);
    auto test_acc_idx = latested_training_log_info.find(test_acc_flag);
    auto end_idx = latested_training_log_info.find(end_flag);

    auto epoch_str = latested_training_log_info.substr(
                         epoch_idx + epoch_flag.size() + 1, time_idx - epoch_idx - epoch_flag.size() - 1
                     );
    auto train_loss_str = latested_training_log_info.substr(
                              train_loss_idx + train_loss_flag.size() + 1,
                              test_loss_idx - train_loss_idx - train_loss_flag.size() - 1
                          );
    auto test_loss_str = latested_training_log_info.substr(
                             test_loss_idx + test_loss_flag.size() + 1,
                             train_acc_idx - test_loss_idx - test_loss_flag.size() - 1
                         );
    auto train_acc_str = latested_training_log_info.substr(
                             train_acc_idx + train_acc_flag.size() + 1,
                             test_acc_idx - train_acc_idx - train_acc_flag.size() - 1
                         );
    auto test_acc_str = latested_training_log_info.substr(
                            test_acc_idx + test_acc_flag.size() + 1,
                            end_idx - test_acc_idx - test_acc_flag.size() - 1
                        );

    *epoch = std::atoi(epoch_str.c_str());
    *train_loss = std::atof(train_loss_str.c_str());
    *test_loss = std::atof(test_loss_str.c_str());
    *train_acc = std::atof(train_acc_str.c_str());
    *test_acc = std::atof(test_acc_str.c_str());
    return true;
}

    inline bool MonitorUtils::_is_process_alive(const std::string &process_name) {
        FILE* fp = nullptr;
        int count = 0;
        int buf_size = 100;
        char buf[buf_size];
        char command_buf[128];
        sprintf(command_buf, "ps -ef | grep -w %s | wc -l", process_name.c_str());

        if ((fp = popen(command_buf, "r")) == nullptr) {
            LOG(ERROR) << "popen err";
            return false;
        }
        if ((fgets(buf, buf_size, fp)) != nullptr) {
            count = atoi(buf);
        }
        pclose(fp);
        fp = nullptr;
        if (count <= 2) {
            return false;
        } else {
            return true;
        }
    }

}
}