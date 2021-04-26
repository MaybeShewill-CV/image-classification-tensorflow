// Copyright 2021 MaybeShewill-CV All Rights Reserved..
// Author: MaybeShewill-CV
// File: tf_utls.h
// Date: 2021/4/26 下午2:03

#ifndef WORKFLOW_MONITOR_TF_UTLS_H
#define WORKFLOW_MONITOR_TF_UTLS_H

#include <string>
#include <iostream>

#include <glog//logging.h>

#include "file_system_utils//file_system_processor.h"

namespace wf_monitor {
    namespace utils {

        using beec::common::file_system_utils::FileSystemProcessor;

        /***
         *
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
            checkpoint_file.open(check_point_record_file_path, std::ios::in);
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
         * @param checkpoint_model_name
         * @return
         */
        bool is_checkpoint_model_evaluated(const std::string& eval_log_file_path,
                                           const std::string& checkpoint_model_name) {
            std::ifstream eval_file;
            eval_file.open(eval_log_file_path, std::ios::in);
            if (!eval_file.is_open() || !eval_file.good()) {
                LOG(ERROR) << "Open evaluation record file: " << eval_log_file_path << ", failed";
                return false;
            }

            std::string record_info;
            bool model_has_been_evaluated = false;
            while (std::getline(eval_file, record_info)) {
                if (record_info.find(checkpoint_model_name) != std::string::npos) {
                    model_has_been_evaluated = true;
                    break;
                }
            }
            eval_file.close();
            return model_has_been_evaluated;
        }

        /***
         * get checkpoint model eval statics
         * @param eval_log_file_path
         * @param checkpoint_model_name
         * @param dataset_name
         * @param dataset_flag
         * @param image_count
         * @param precision
         * @param recall
         * @param f1
         * @return
         */
        bool get_checkpoint_model_eval_statics(
                const std::string& eval_log_file_path,
                const std::string& checkpoint_model_name,
                std::string& dataset_name,
                std::string& dataset_flag,
                int32_t image_count, float_t precision, float_t recall, float_t f1) {
            if (!is_checkpoint_model_evaluated(eval_log_file_path, checkpoint_model_name)) {
                return false;
            }

            std::ifstream eval_file;
            eval_file.open(eval_log_file_path, std::ios::in);
            if (!eval_file.is_open() || !eval_file.good()) {
                LOG(ERROR) << "Open evaluation record file: " << eval_log_file_path << ", failed";
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
                    image_count = std::atoi(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    // read model name
                    std::getline(eval_file, tmp_info);
                    std::string model_name = tmp_info.substr(tmp_info.find_last_of(':') + 1);
                    // read model precision
                    std::getline(eval_file, tmp_info);
                    precision = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    // read model recall
                    std::getline(eval_file, tmp_info);
                    recall = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    // read model f1
                    std::getline(eval_file, tmp_info);
                    f1 = std::atof(tmp_info.substr(tmp_info.find_last_of(':') + 1).c_str());
                    break;
                }
            }
            eval_file.close();
            return true;
        }
    }
}

#endif //WORKFLOW_MONITOR_TF_UTLS_H
