// Copyright 2021 MaybeShewill-CV All Rights Reserved..
// Author: MaybeShewill-CV
// File: tf_utls.h
// Date: 2021/4/26 下午2:03

#ifndef WORKFLOW_MONITOR_TF_UTLS_H
#define WORKFLOW_MONITOR_TF_UTLS_H

#include <string>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

#include <glog/logging.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include "file_system_utils//file_system_processor.h"

namespace wf_monitor {
    namespace utils {

        using beec::common::file_system_utils::FileSystemProcessor;
        namespace fs = boost::filesystem;

        /***
         * get latest checkpoint model file path
         * @param model_dir
         * @return
         */
        bool get_latest_checkpoint(const std::string& model_dir, std::string& model_checkpoint_path) {
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

        /***
         * check if the checkpoint model has been evaluated
         * @param eval_log_file_path
         * @param model_name
         * @return
         */
        bool is_checkpoint_model_evaluated(const std::string& eval_log_file_path,
                                           const std::string& model_name) {
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

        /***
         * judge if a process is alive
         * @param command
         * @return
         */
        bool is_process_alive(const std::string& process_name) {
            FILE* fp = nullptr;
            int count = 1;
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
            if (count <= 1) {
                LOG(INFO) << "No active process for: " << process_name;
                return false;
            } else {
                return true;
            }
        }

        /***
         * judge if the training process is alive
         * @return
         */
        bool is_net_training_process_alive() {
            return is_process_alive("train_model.py");
        }

        /***
         * judge if the evaluating process is alive
         * @return
         */
        bool is_net_evaluating_process_alive() {
            return is_process_alive("evaluate_model.py");
        }

        /***
         *
         * @param training_log_files
         * @return
         */
        bool get_all_training_log_files(const std::string& log_dir, std::vector<std::string>& training_log_files) {
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

        /***
         *
         * @param file_path
         * @return
         */
        std::time_t get_file_last_modified_time(const std::string& file_path) {
            if (!FileSystemProcessor::is_file_exist(file_path)) {
                return 0;
            }
            struct stat buf;
            FILE *pFile = nullptr;
            pFile = fopen(file_path.c_str(), "r");
            int fd = fileno(pFile);
            fstat(fd, &buf);
            std::time_t time = buf.st_mtime;
            return time;
        }

        /***
         *
         * @param log_dir
         * @param latested_log_file_path
         * @return
         */
        bool get_latested_training_log_file(const std::string& log_dir, std::string& latested_log_file_path) {
            std::vector<std::string> tmp_log_file_paths;
            if (!get_all_training_log_files(log_dir, tmp_log_file_paths)) {
                latested_log_file_path = "";
                return false;
            }
            std::map<std::time_t, std::string> tmp_log_file_map;
            for (size_t index = 0; index < tmp_log_file_paths.size(); ++index) {
                std::string tmp_file_path = tmp_log_file_paths[index];
                tmp_log_file_map.insert(
                        std::make_pair(get_file_last_modified_time(tmp_file_path), tmp_file_path));
            }
            latested_log_file_path = tmp_log_file_map.rbegin()->second;
            return true;
        }

        /***
         * get model's name which is been training now
         * @param log_dir
         * @param model_name
         * @return
         */
        bool get_training_model_name(const std::string& log_dir, std::string& model_name) {
            if (!is_net_training_process_alive()) {
                LOG(INFO) << "Get training model name failed";
                model_name = "";
                return false;
            }

            std::string latest_log_file_path;
            if (!get_latested_training_log_file(log_dir, latest_log_file_path)) {
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

        /***
         * get dataset's name which is been training now
         * @param log_dir
         * @param dataset_name
         * @return
         */
        bool get_training_dataset_name(const std::string& log_dir, std::string& dataset_name) {
            if (!is_net_training_process_alive()) {
                LOG(INFO) << "Get training model name failed";
                dataset_name = "";
                return false;
            }

            std::string latest_log_file_path;
            if (!get_latested_training_log_file(log_dir, latest_log_file_path)) {
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

        /***
         * get checkpoint mode save dir
         * @param project_base_dir
         * @return
         */
        bool get_checkpoint_model_save_dir(const std::string& project_base_dir, std::string& model_save_dir) {

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

        /***
         * get eval log file path
         * @param log_dir
         * @param eval_log_file_path
         * @return
         */
        bool get_eval_log_file_path(const std::string& log_dir, std::string& eval_log_file_path) {
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

        /***
         * get checkpoint model eval statics from eval log file
         * @param eval_log_file_path
         * @param model_name
         * @param checkpoint_model_name
         * @param dataset_flag
         * @param image_count
         * @param precision
         * @param recall
         * @param f1
         * @return
         */
        bool _get_checkpoint_model_eval_statics_impl(
                const std::string& eval_log_file_path,
                const std::string& checkpoint_model_name,
                std::string& dataset_name,
                std::string& dataset_flag,
                int32_t* image_count, float_t* precision, float_t* recall, float_t* f1) {
            if (!is_checkpoint_model_evaluated(eval_log_file_path, checkpoint_model_name)) {
                dataset_name = "";
                dataset_flag = "";
                *image_count = 0;
                *precision = 0.0;
                *recall = 0.0;
                *f1 = 0.0;
                return false;
            }

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

            std::string record_info;
            while (std::getline(eval_file, record_info)) {
                if (record_info.find(checkpoint_model_name) != std::string::npos) {
                    std::string tmp_info;
                    // read dataset name
                    std::getline(eval_file, tmp_info);
                    dataset_name = tmp_info.substr(tmp_info.find_last_of(':') + 1);
                    // read dataset flag
                    std::getline(eval_file, tmp_info);
                    dataset_flag = tmp_info.substr(tmp_info.find_last_of(':') + 1);
                    // read dataset image count
                    std::getline(eval_file, tmp_info);
                    *image_count = std::atoi(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    // read model name
                    std::getline(eval_file, tmp_info);
                    std::string model_name = tmp_info.substr(tmp_info.find_last_of(':') + 1);
                    // read model precision
                    std::getline(eval_file, tmp_info);
                    *precision = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    // read model recall
                    std::getline(eval_file, tmp_info);
                    *recall = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    // read model f1
                    std::getline(eval_file, tmp_info);
                    *f1 = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    break;
                }
            }
            eval_file.close();
            return true;
        }

        /***
         *
         * @param project_dir
         * @param dataset_name
         * @param dataset_flag
         * @param image_count
         * @param precision
         * @param recall
         * @param f1
         * @return
         */
        bool get_checkpoint_model_eval_statics(
                const std::string& project_dir, std::string& dataset_name,
                std::string& dataset_flag,
                int32_t* image_count, float_t* precision, float_t* recall, float_t* f1) {
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
            // get checkpoint model name
            std::string model_dir;
            if (!get_checkpoint_model_save_dir(project_dir, model_dir)) {
                LOG(ERROR) << "Get model eval statics failed";
                dataset_name = "";
                dataset_flag = "";
                *image_count = 0;
                *precision = 0.0;
                *recall = 0.0;
                *f1 = 0.0;
                return false;
            }
            std::string checkpoint_model_path;
            if (!get_latest_checkpoint(model_dir, checkpoint_model_path)) {
                LOG(ERROR) << "Get model eval statics failed";
                dataset_name = "";
                dataset_flag = "";
                *image_count = 0;
                *precision = 0.0;
                *recall = 0.0;
                *f1 = 0.0;
                return false;
            }
            std::string checkpoint_model_name = FileSystemProcessor::get_file_name(checkpoint_model_path);
            return _get_checkpoint_model_eval_statics_impl(
                    eval_log_file_path, checkpoint_model_name, dataset_name,
                    dataset_flag, image_count, precision, recall, f1);
        }

        /***
         * Get training status from training log file
         * @param trainning_log_file_path
         * @param epoch
         * @param train_loss
         * @param test_loss
         * @param train_acc
         * @param test_acc
         * @return
         */
        bool _get_model_training_statics_impl(
                const std::string& trainning_log_file_path,
                int* epoch, float* train_loss, float* test_loss, float* train_acc, float* test_acc) {
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

        /***
         *
         * @param project_dir
         * @param epoch
         * @param train_loss
         * @param test_loss
         * @param train_acc
         * @param test_acc
         * @return
         */
        bool get_model_training_statics(
                const std::string& project_dir,
                int* epoch, float* train_loss, float* test_loss, float* train_acc, float* test_acc) {
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
            if (!get_latested_training_log_file(training_log_dir, train_log_file_path)) {
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
    }
}

#endif //WORKFLOW_MONITOR_TF_UTLS_H
